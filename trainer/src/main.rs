use std::env;
use std::process::ExitCode;

use bullet_lib::game::inputs::{Chess768, ChessBucketsMirrored, get_num_buckets};
use bullet_lib::game::outputs::MaterialCount;
use bullet_lib::nn::optimiser::{AdamW, AdamWParams};
use bullet_lib::nn::{InitSettings, Shape};
use bullet_lib::trainer::save::SavedFormat;
use bullet_lib::trainer::schedule::{TrainingSchedule, TrainingSteps, lr, wdl};
use bullet_lib::trainer::settings::LocalSettings;
use bullet_lib::value::loader::sfbinpack::{
    MoveType, PieceType, SfBinpackLoader, TrainingDataEntry,
};
use bullet_lib::value::{ValueTrainerBuilder, loader};

// These constants are part of the network file's contract with the engine's inference code; the hidden size is
// the other part and is given on the command line.
const SCALE: i32 = 400;
const QA: i16 = 255;
const QB: i16 = 64;

const BATCH_SIZE: usize = 16_384;
// A binpack superbatch is 6,104 batches, about 100 million positions.
const BINPACK_BATCHES: usize = 6_104;

// The bucketed network's layout, also part of the contract with `engine/src/nnue.rs`: input weights chosen by the
// own king's square, mirrored so the king stands on files a to d (one entry per rank and mirrored file), and output
// weights chosen by the number of pieces on the board.
#[rustfmt::skip]
const KING_BUCKETS: [usize; 32] = [
    0, 0, 1, 1,
    2, 2, 2, 2,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3,
];
const INPUT_BUCKETS: usize = get_num_buckets(&KING_BUCKETS);
const OUTPUT_BUCKETS: usize = 8;

const USAGE: &str = "usage: lk-47-trainer text <file> <positions> <passes> <hidden> <wdl> <out>\n       \
                     lk-47-trainer binpack <file> <superbatches> <hidden> <wdl> <out>\n       \
                     lk-47-trainer buckets <file[,file...]> <superbatches> <hidden> <wdl> <out>";

// Quiet positions only, past the opening, as in bullet's own binpack example.
fn quiet(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    let number = |text: &str| text.parse::<usize>().ok();
    if let ["buckets", file, superbatches, hidden, wdl, out] = words.as_slice() {
        return match (number(superbatches), number(hidden), wdl.parse::<f32>()) {
            (Some(superbatches), Some(hidden), Ok(wdl)) => {
                train_bucketed(file, superbatches, hidden, wdl, out);
                ExitCode::SUCCESS
            }
            _ => {
                eprintln!("{USAGE}");
                ExitCode::FAILURE
            }
        };
    }
    let (binpack, file, batches, superbatches, hidden, wdl, out) = match words.as_slice() {
        ["text", file, positions, passes, hidden, wdl, out] => {
            match (
                number(positions),
                number(passes),
                number(hidden),
                wdl.parse::<f32>(),
            ) {
                (Some(positions), Some(passes), Some(hidden), Ok(wdl)) => (
                    false,
                    *file,
                    positions.div_ceil(BATCH_SIZE),
                    passes,
                    hidden,
                    wdl,
                    *out,
                ),
                _ => {
                    eprintln!("{USAGE}");
                    return ExitCode::FAILURE;
                }
            }
        }
        ["binpack", file, superbatches, hidden, wdl, out] => {
            match (number(superbatches), number(hidden), wdl.parse::<f32>()) {
                (Some(superbatches), Some(hidden), Ok(wdl)) => (
                    true,
                    *file,
                    BINPACK_BATCHES,
                    superbatches,
                    hidden,
                    wdl,
                    *out,
                ),
                _ => {
                    eprintln!("{USAGE}");
                    return ExitCode::FAILURE;
                }
            }
        }
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, hidden);
            let l1 = builder.new_affine("l1", 2 * hidden, 1);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden))
        });

    // The learning rate drops tenfold at three quarters of the run; four checkpoints are saved along the way.
    let schedule = TrainingSchedule {
        net_id: "lk47".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: batches,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl },
        lr_scheduler: lr::StepLR {
            start: 0.001,
            gamma: 0.1,
            step: (superbatches * 3 / 4).max(1),
        },
        save_rate: (superbatches / 4).max(1),
    };
    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: out,
        batch_queue_size: 64,
    };
    if binpack {
        trainer.run(
            &schedule,
            &settings,
            &SfBinpackLoader::new(file, 1024, 4, quiet),
        );
    } else {
        trainer.run(
            &schedule,
            &settings,
            &loader::DirectSequentialDataLoader::new(&[file]),
        );
    }
    ExitCode::SUCCESS
}

// King-bucketed inputs with a shared factoriser merged in on saving, and material-count output buckets saved one
// bucket's weights after another. The factoriser and the bucket weights are each clipped to 0.99, so their sum keeps
// the 1.98 bound that the engine's 16-bit arithmetic relies on.
fn train_bucketed(file: &str, superbatches: usize, hidden: usize, wdl: f32, out: &str) {
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(ChessBucketsMirrored::new(KING_BUCKETS))
        .output_buckets(MaterialCount::<OUTPUT_BUCKETS>)
        .save_format(&[
            SavedFormat::id("l0w")
                .transform(|store, weights| {
                    let factoriser = store.get("l0f").values.f32().repeat(INPUT_BUCKETS);
                    weights
                        .into_iter()
                        .zip(factoriser)
                        .map(|(a, b)| a + b)
                        .collect()
                })
                .round()
                .quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w")
                .round()
                .quantise::<i16>(QB)
                .transpose(),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs, output_buckets| {
            let l0f = builder.new_weights("l0f", Shape::new(hidden, 768), InitSettings::Zeroed);
            let mut l0 = builder.new_affine("l0", 768 * INPUT_BUCKETS, hidden);
            l0.weights = l0.weights + l0f.repeat(INPUT_BUCKETS);
            let l1 = builder.new_affine("l1", 2 * hidden, OUTPUT_BUCKETS);
            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden))
                .select(output_buckets)
        });
    let clip = AdamWParams {
        max_weight: 0.99,
        min_weight: -0.99,
        ..Default::default()
    };
    trainer.optimiser.set_params_for_weight("l0w", clip);
    trainer.optimiser.set_params_for_weight("l0f", clip);

    // The learning rate falls from 0.001 to 0.001 * 0.3^5 along a cosine; four checkpoints are saved along the way.
    let schedule = TrainingSchedule {
        net_id: "lk47".to_string(),
        eval_scale: SCALE as f32,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BINPACK_BATCHES,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: wdl },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: 0.001,
            final_lr: 0.001 * 0.3f32.powi(5),
            final_superbatch: superbatches,
        },
        save_rate: (superbatches / 4).max(1),
    };
    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: out,
        batch_queue_size: 64,
    };
    trainer.run(
        &schedule,
        &settings,
        &SfBinpackLoader::new_concat_multiple(&file.split(',').collect::<Vec<_>>(), 1024, 4, quiet),
    );
}
