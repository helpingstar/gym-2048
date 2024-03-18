from setuptools import setup

setup(
    name="gym-game2048",
    version="0.1.0",
    author="helpingstar",
    author_email="iamhelpingstar@gmail.com",
    description="A reinforcement learning environment for the 2048 game based on Gymnasium.",
    license="MIT License",
    install_requires=["numpy>=1.21.0", "gymnasium>=1.0.0a1", "moviepy>=1.0.0", "pygame>=2.1.3"],
    python_requires=">=3.9",
)
