from setuptools import find_packages, setup

setup(
    name="gen_ai",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={"console_scripts": ["gen_ai_train=gen_ai:train", "gen_ai_sample=gen_ai:sample"]},
)
