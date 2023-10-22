# Getting Started

## Installation
### From PyPI
Flicker is available on [PyPI](https://pypi.org/project/flicker/). You can install it using `pip`.

```bash
pip install flicker
```

### Building from source
This may be needed if you want to develop and contribute code.

#### Clone the Repository
```bash
git clone git@github.com:ankur-gupta/flicker.git
```

#### Install `hatch`
`flicker` uses [`hatch`](https://hatch.pypa.io/latest/) build system. Best way to 
[install `hatch`](https://hatch.pypa.io/latest/install/#pipx) is via [`pipx`](https://github.com/pypa/pipx). 
```bash
pipx install hatch
```

#### Build `flicker`
```bash
hatch run test:with-coverage
hatch build
```

#### Install wheel via `pip`
```bash
pip install $REPO_ROOT/dist/flicker-x.y.z-py3-none-any.whl
```

