# Magnetic Fusion Performance

In this tutorial you'll learn how to predict the performance for fusion devices known as tokamaks. In magnetic confinement fusion, strong magnetic fields are used to to ensure the heat in a 150 million °C fusion plasma is not quickly lost to the surrounding environment.

The rate of this energy loss is measured by the energy confinement time, τ. A high confinement time corresponds to a better-performing device, and vice versa. If we have accurate ways of predicting confinement using the design features of a fusion device, then we can better optimise the fusion power-plants of the future.

## Quick start

Clone the repository and change directory to the project root:

```
git clone https://github.com/digiLab-ai/Magnetic-Fusion-Performance.git
cd Magnetic-Fusion-Performance
```

Install the dependencies:

```
poetry install
```

Enter your API key into the relevant cell in the notebook, or copy the .env.example file to .env

```
cp .env.example .env
```

and fill out your twinLab login details in .env

Run the demo notebook:

```
poetry run jupyter notebook demo.ipynb
```
