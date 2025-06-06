# import os
# import importlib
# import pkgutil
#
# # import all modules in the dataset package to register the datasets
# for _, module_name, _ in pkgutil.iter_modules([os.path.dirname("dataset")]):
#     if module_name == "dataset":
#         for _, sub_module_name, _ in pkgutil.iter_modules([module_name]):
#             importlib.import_module(f".{sub_module_name}", package=__name__)
