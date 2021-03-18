from setuptools import find_packages, setup

import src.imitation  # pytype: disable=import-error

TESTS_REQUIRE = [
    "seals>=0.1.1",
    "black",
    # remove pin once https://github.com/nedbat/coveragepy/issues/881 fixed
    "coverage==4.5.4",
    "codecov",
    "codespell",
    "flake8",
    "flake8-blind-except",
    "flake8-builtins",
    "flake8-debugger",
    "flake8-isort",
    "pytest",
    "pytest-cov",
    "pytest-notebook",
    "pytest-xdist",
    "pytype",
]
DOCS_REQUIRE = [
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx-rtd-theme",
    "sphinxcontrib-napoleon",
]
PARALLEL_REQUIRE = ["ray[debug,tune]~=0.8.5"]


def get_readme() -> str:
    """Retrieve content from README."""
    with open("README.md", "r") as f:
        return f.read()


setup(
    name="imitation",
    version=src.imitation.__version__,
    description="Implementation of modern IRL and imitation learning algorithms.",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    author="Center for Human-Compatible AI and Google",
    python_requires=">=3.6.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"imitation": ["py.typed", "envs/examples/airl_envs/assets/*.xml"]},
    install_requires=[
        "awscli",
        "cloudpickle>=0.5.5",
        "gym[classic_control]",
        "matplotlib",
        "numpy>=1.15",
        "torch>=1.4.0",
        "tqdm",
        "scikit-learn>=0.21.2",
        "stable-baselines3~=0.10.0",
        "jax~=0.2.8",
        "jaxlib~=0.1.59",
        "sacred~=0.8.1",
        "tensorboard>=1.14",
    ],
    tests_require=TESTS_REQUIRE,
    extras_require={
        # recommended packages for development
        "dev": [
            "autopep8",
            "ntfy[slack]",
            "ipdb",
            "isort~=5.0",
            "jupyter",
            "pytype",
            "codespell",
            # for convenience
            *TESTS_REQUIRE,
            *DOCS_REQUIRE,
        ],
        "test": TESTS_REQUIRE,
        "docs": DOCS_REQUIRE,
        "parallel": PARALLEL_REQUIRE,
    },
    entry_points={
        "console_scripts": [
            ("imitation-expert-demos=imitation.scripts.expert_demos" ":main_console"),
            "imitation-train=imitation.scripts.train_adversarial:main_console",
        ],
    },
    url="https://github.com/HumanCompatibleAI/imitation",
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)
