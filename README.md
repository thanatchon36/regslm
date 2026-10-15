# regslm

Small language models that run entirely on-premises and turn public financial rulebooks into
structured **obligations** — who must do what, under which condition, with which
threshold/deadline, citing which paragraph — plus a pre-registered study of how model size,
adaptation method, label budget and quantization trade off.

> Status: under construction (Phase 1). See `docs/PROGRESS.md`.
>
> Independent project; public documents only; not legal, regulatory or compliance advice.
> Output must be reviewed by a qualified person. See [DISCLAIMER.md](DISCLAIMER.md).

```bash
uv sync
uv run regslm doctor
```

Licence: Apache-2.0 (code). Corpus text is under per-source terms — see
`docs/sources/LICENSES.md`.
