import os
import importlib
import pkgutil


# import all modules in the modeling package to register the models
for _, module_name, _ in pkgutil.iter_modules():
    if module_name == "modeling":
        for _, sub_module_name, _ in pkgutil.iter_modules([module_name]):
            importlib.import_module(f".{sub_module_name}", package="modeling")
    if module_name == "dataset":
        for _, sub_module_name, _ in pkgutil.iter_modules([module_name]):
            importlib.import_module(f".{sub_module_name}", package="dataset")

