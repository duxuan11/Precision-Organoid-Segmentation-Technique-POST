
import importlib

def check_requirements(package_names):
    """
    Raise error if module is not installed.
    """
    missing_packages = []
    for package_name in package_names:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)
    if missing_packages:
        raise ImportError(f"The following packages are required to use this module: {missing_packages}")
    yield