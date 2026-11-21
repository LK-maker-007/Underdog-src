# Underdog-src

Source of LK-47, a UCI chess engine.

Prebuilt binaries are on the [releases page](https://github.com/LK-maker-007/Underdog/releases).

## Layout

- `engine/` — UCI engine: search, move generation, NNUE evaluation, threading
- `trainer/` — NNUE network training

## Build

```
cargo build --release
```

The binary lands in `target/release/`.

## Usage

LK-47 speaks UCI, so it runs under any UCI GUI (Cute Chess, Arena, BanksiaGUI) or directly:

```
uci
isready
position startpos
go depth 12
```

## License

MIT — see [LICENSE](LICENSE).
