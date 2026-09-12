# vendor/

Third-party dependencies managed as [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

| Directory | Upstream | Description |
|-----------|----------|-------------|
| `midstream/` | [ruvnet/midstream](https://github.com/ruvnet/midstream) | Claude Flow middleware and agent orchestration |
| `ruvector/` | [ruvnet/ruvector](https://github.com/ruvnet/ruvector) | RuVector signal processing and ML pipelines |
| `sublinear-time-solver/` | [ruvnet/sublinear-time-solver](https://github.com/ruvnet/sublinear-time-solver) | Sublinear-time optimization solvers |

All submodules track the `main` branch of their upstream repos.

## Setup

After cloning this repo, initialize submodules:

```bash
git submodule update --init --recursive
```

Or clone with submodules in one step:

```bash
git clone --recurse-submodules https://github.com/ruvnet/RuView.git
```

## Update to latest upstream

```bash
git submodule update --remote --recursive
git add $(git config --file .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')
git commit -m "chore: update vendor submodules"
```

A GitHub Actions workflow also checks for updates every 6 hours. It reuses the
`chore/update-submodules` branch and opens or updates a PR automatically when
repository settings allow GitHub Actions to create pull requests.
