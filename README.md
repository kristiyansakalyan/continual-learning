# continual-learning
A research project investigating catastrophic forgetting in DNNs for medical data.

## Datasets
The datasets we use are Cadis and frames from the Cataract-101 dataset. 
You can download them both [here](https://ftp.itec.aau.at/datasets/ovid/InSegCat/index.html).

They are both in the COCO-format. You can learn more about it [here](https://cocodataset.org/#format-data).

## Code related
Some important notes regarding package management and code quality.


### Managing dependencies
We can use poetry as a package manager and make sure that we all have the same dependencies. Here is a small tutorial on how to set it up:
```
python -m pip install pipx
pipx install poetry
pipx ensurepath
```
Done! You now have poetry. Now you can simply do:
```
poetry install
```
You're good to code now. If you want to add or remove packages, simply use:
```
poetry add <package_name>
poetry remove <package_name>
```
### Pre-commit hook
Now that you have installed all dependencies with poetry, we can setup a pre-commit hook, so that our code is always formatted and sorted. (Later on we can add ruff, mypy and others in the pre-commit hook). You can install it with:
```
pre-commit install
```
If you want to run it before commiting, you can do:
```
pre-commit run --all-files
```
