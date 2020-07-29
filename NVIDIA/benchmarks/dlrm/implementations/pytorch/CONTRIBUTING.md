# Contributing guidelines

## Style

#### C/C++ coding style

Follow [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html), with some exceptions:

* max-line-length extended to 120.
* Allow mixed case name, . e.g. `tableName`. Google style prohibits it. But it is so widely used in NVIDIA, I concede.
* Allow C type cast, e.g. `(float *)foo`

Run `cpplint` before committing code.

#### Python coding style

Follow [Google Python Style Guide,](https://google.github.io/styleguide/pyguide.html) with some exceptions:

* max-line-length extended to 120. 
* Docstring `Args` follows Facebook Python style, e.g. type in a bracket.
* Exceptions are allowed if it feels more natural to follow Facebook style. For example, Pytorch allows import relative path, also class name.
* Allow names used in Pytorch internal that violate rules, e.g. `x`, `input`, `dX`...

Run `pylint` before committing code. It doesn't mean every issue has to be corrected nor check has to be manually disabled. Just make sure you are aware of the remaining issues and you are comfort with all of them. 

Install `pylint`

```bash
pip install pylint
```

To check a file with `pylint`:

```bash
pylint --rcfile=.pylintrc myfile.py
```

#### Yapf

[yapf](https://github.com/google/yapf/) is an auto format tool owned by Google (not a Google product). To save the time of arguing code style during code review, use yapf to format the code is a good option. Note that it doesn't reformat comment.

Install `yapf`

```bash
pip install yapf
```

Format code with yapf

```bash
yapf myfile.py --style .style.yapf
```

There are Sublime and Vim plugins.
