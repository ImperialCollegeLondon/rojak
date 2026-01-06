# `rojak`: A Library and CLI Tool for Aviation Turbulence Analysis

[![CI](https://github.com/ImperialCollegeLondon/rojak/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/ImperialCollegeLondon/rojak/actions/workflows/ci.yml)
[![CD](https://github.com/ImperialCollegeLondon/rojak/actions/workflows/cd.yml/badge.svg?branch=master)](https://github.com/ImperialCollegeLondon/rojak/actions/workflows/cd.yml)
[![codecov](https://codecov.io/gh/ImperialCollegeLondon/rojak/graph/badge.svg?token=0COCM07N7R)](https://codecov.io/gh/ImperialCollegeLondon/rojak)

[![status](https://joss.theoj.org/papers/efb0bc9eab2eadeebc1203d15a3203e5/status.svg)](https://joss.theoj.org/papers/efb0bc9eab2eadeebc1203d15a3203e5)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17882683.svg)](https://doi.org/10.5281/zenodo.17882683)

[![python](https://img.shields.io/pypi/pyversions/rojak-cat.svg)](https://github.com/ImperialCollegeLondon/rojak)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)
[![Formatted with ruff](https://img.shields.io/badge/code%20style-ruff-d7ff64)](https://github.com/astral-sh/ruff)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)

`rojak` is a distributed python library and command-line tool for using meterological data to forecast CAT and evaluating the effectiveness of CAT diagnostics against turbulence observations.
Currently, it supports,

1. Computing turbulence diagnostics on meteorological data from [European Centre for Medium-Range Weather Forecasts's (ECMWF) ERA5 reanalysis on pressure levels](https://doi.org/10.24381/cds.bd0915c6). Moreover, it is easily extendable to support other types of meteorological data.
2. Retrieving and processing turbulence observations from Aircraft Meteorological Data Relay (AMDAR) data [archived at National Oceanic and Atmospheric Administration (NOAA)](https://amdar.ncep.noaa.gov/index.shtml) and [AMDAR data collected via the Met Office MetDB system](https://catalogue.ceda.ac.uk/uuid/33f44351f9ceb09c495b8cef74860726/).
3. Computing 27 different turbulence diagnostics, such as the three-dimensional frontogenesis equation, turbulence index 1 and 2, negative vorticity advection, and Brown's Richardson tendency equation.
4. Converting turbulence diagnostic values into the eddy dissipation rate (EDR) --- the International Civil Aviation Organization's (ICAO) official metric for reporting turbulence.

>[!NOTE]
> This repository is under active development. As a result,
>
> 1. The API is subject to change and may not be stable
> 2. The documentation is incomplete and will be updated as development progresses.
>
> We appreciate your understanding and encourage you to check back for updates.

Rojak is also a salad with Javanese origins. Colloquially (in Singlish), it means a mixture of things.
This package is for a traditionally unconventional mixture of aviation meteorology (turbulence diagnostics) and (coming
soon) aeroelasticity.

## What could I use `rojak` for?

If you're an aviation meteorologist, `rojak` can be used to computing the various turbulence diagnostics and converting it into EDR like [this map from the Aviation Weather Center](https://aviationweather.gov/gfa/#turb).

Or if you're a researcher and you'd like to understand how turbulence is impacted by climate change, `rojak` can be used to compute the probability of encountering turbulence and examining how that has changed over time.

## Documentation

Learn more about `rojak` at [imperialcollegelondon.github.io/rojak/](https://imperialcollegelondon.github.io/rojak/).

## Installation

 For installation instructions, please see the [Installation Guide](https://imperialcollegelondon.github.io/rojak/userguide/installguide.html#installation).
