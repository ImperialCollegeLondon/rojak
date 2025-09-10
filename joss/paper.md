---
title: "rojak: A Python library and tool for aviation turbulence diagnostics"
tags:
  - Python
  - Clear air turbulence
  - Aviation
authors:
  - given-names: Hui Ling
    surname: Wong
    orcid: 0009-0006-2314-2660
    corresponding: true
    affiliation: "1, 2"
  - name: Rafael Palacios
    affiliation: 1
    orcid: 0000-0002-6706-3220
  - name: Edward Gryspeerdt
    orcid: 0000-0002-3815-4756
    affiliation: 2
affiliations:
  - index: 1
    name: Department of Aeronautics, Imperial College London, United Kingdom
    ror: 041kmwe10
  - index: 2
    name: Department of Physics, Imperial College London, United Kingdom
    ror: 041kmwe10
date: 22 July 2025
bibliography: paper.bib
---

# Summary

Aviation turbulence is atmospheric turbulence occurring at length scales large enough (approximately 100m to 1km) to affect an aircraft [@sharmanNatureAviationTurbulence2016]. According to the National Transport Safety Board (NTSB), turbulence experienced whilst onboard an aircraft was the leading cause of accidents from 2009 to 2018 [@ntsbPreventingTurbulenceRelatedInjuries2021].
Clear air turbulence (CAT) is a form of aviation turbulence which cannot be detected by the onboard weather radar. Thus, pilots are unable to preemptively avoid such regions.
In order to mitigate this safety risk, CAT diagnostics are used to forecast turbulent regions such that pilots are able to tactically avoid them.

`rojak` is a parallelised python library and command-line tool for using meteorological data to forecast CAT and evaluating the effectiveness of CAT diagnostics against turbulence observations.
Currently, it supports,

1. Computing turbulence diagnostics on meteorological data from European Centre for Medium-Range Weather Forecasts's (ECMWF) ERA5 reanalysis on pressure levels [@hersbach2023era5]. Moreover, it is easily extendable to support other types of meteorological data.
2. Retrieving and processing turbulence observations from Aircraft Meteorological Data Relay (AMDAR) data archived at National Oceanic and Atmospheric Administration (NOAA)[@NCEPMeteorologicalAssimilation] and AMDAR data collected via the Met Office MetDB system [@ukmoAmdar]
3. Computing 27 different turbulence diagnostics, such as the three-dimensional frontogenesis equation [@bluesteinSynopticdynamicMeteorologyMidlatitudes1993], turbulence index 1 and 2 [@ellrodObjectiveClearAirTurbulence1992], negative vorticity advection [@sharmanIntegratedApproachMid2006], and Brown's Richardson tendency equation [@brownNewIndicesLocate1973].
4. Converting turbulence diagnostic values into the eddy dissipation rate (EDR) --- the International Civil Aviation Organization's (ICAO) official metric for reporting turbulence [@icaoAnnex3]

These features not only allow users to perform operational forecasting of CAT but also to interrogate the intensification in frequency and severity of CAT due to climate change [@williamsIncreasedLightModerate2017; @storerGlobalResponseClearAir2017; @kimGlobalResponseUpperlevel2023], such as by analysing the climatological distribution of the probability of encountering turbulence at different severities (e.g. light turbulence or moderate-or-greater turbulence) for each turbulence diagnostic.
These applications involve high-volume datasets, ranging from tens to hundreds of gigabytes, necessitating the use of parallelisation to preserve computational tractability and efficiency, while substantially reducing execution time. As such, `rojak` leverages `Dask` to process larger-than-memory data and to run in a distributed manner [@pydataDask].

The name of the package, `rojak`, is inspired by its wide range turbulence diagnostics and its applications. While _rojak_ refers to a type of salad, it is also a colloquial term in Malaysia and Singapore for an eclectic mix, reflecting the diverse functionality of the package.

# Statement of need


Numerous studies have investigated the influence of the climate on CAT [e.g. @kimGlobalResponseUpperlevel2023; @williamsIncreasedLightModerate2017; @storerGlobalResponseClearAir2017] and the application of turbulence diagnostics for operational forecasting [e.g. @gillObjectiveVerificationWorld2014; @sharmanIntegratedApproachMid2006; @sharmanPredictionEnergyDissipation2017; @gillEnsembleBasedTurbulence2014] in products like the Federal Aviation Authority's (FAA) Graphical Turbulence Guidance and the International Civil Aviation Organization's (ICAO) World Area Forecast System.
However, to the best of the author's knowledge, none of these studies have made their code publicly available. This work presents the first open-source package for aviation turbulence analysis.
Given the inherent complexity of CAT diagnostics and the variability in how these could be implemented, `rojak` serves as a first iteration of a standardised implementation of these CAT diagnostics, providing a basis for future enhancements and refinements by the broader research community.
Moreover, the parallelised nature of `rojak` and its architecture, which keeps it open to extensions, positions it as as an indispensable resource to bridging this gap.

![Probability of encountering light turbulence during the months December, January, February from 2018 to 2024 at 200 hPa for the three-dimensional frontogenesis (F3D) and turbulence index 1 (TI1) diagnostics \label{fig:probability_light_turbulence}](multi_diagnostic_f3d_ti1_on_200_light.png)

\autoref{fig:probability_light_turbulence} demonstrates the application of `rojak` for characterising CAT's response to climate change.
Depicted in the figure is the global climatological distribution of the probability of encountering light turbulence for the boreal winter months (i.e., December, January and February) from 2018 to 2024 at 200 hPa based on the two turbulence diagnostics --- the three-dimensional frontogenesis equation and turbulence index 1. This was computed using ERA5 data at 6-hourly intervals with three pressure levels (175 hPa, 200 hPa and 225 hPa) for the aforementioned time period.
This required processing 85GB of ERA5 data.
The methodology employed by `rojak` for determining the presence of turbulence and the equations for the various turbulence diagnostics is derived from the existing aviation meteorology literature on turbulence.
In this instance, it is an implementation of the methodology described in @williamsCanClimateModel2022.

![6-hour forecast of eddy dissipation rate (EDR) at 200 hPa for the three-dimensional frontogenesis (F3D) and turbulence index 1 (TI1) on the 1st of December 2024 at 00:00 \label{fig:edr}](multi_edr_f3d_ti1.png)

Similarly, \autoref{fig:edr} demonstrates the application of `rojak` for operational turbulence forecasting.
By employing the methodology described in @sharmanPredictionEnergyDissipation2017, `rojak` is able to convert three-dimensional frontogenesis and turbulence index 1 diagnostics into EDR.
\autoref{fig:edr} shows the 6-hour turbulence forecast on the 1st of December 2024 at 00:00 GMT which can be used for flight trajectory planning to avoid turbulent regions.
The full range of features, including the details and references, are contained within the documentation of `rojak`.

In the context of operational forecasting [e.g. methods detailed in @gillObjectiveVerificationWorld2014; @sharmanIntegratedApproachMid2006; and @pearsonPredictionEnergyDissipation2017], the comparison of turbulence diagnostics computed from meteorological data against observational data is a fundamental component. The statistical nature of using an ensemble of turbulence diagnostics which has an optimal balance of a low false positive and false negative rate mandates it.
The architecture of `rojak` enables it to seamlessly integrate various sources of meteorological and observational data, with their interactions managed through a central mediator.
`rojak` also contains a command-line interface tool which can be launched to perform a variety of aviation turbulence analyses and to retrieve the meteorological and observation data from the various data providers.

In terms of the calculations performed upon the meteorological data, `MetPy` [@mayMetPyPythonPackage2016] has the greatest similarity to `rojak`. However, it does not natively support `Dask` [@manserSupportDaskArrays]. Given the size of the datasets to be processed, this presented a significant issue.
Moreover, `MeyPy` does not implement the calculations required by the turbulence diagnostics.


# Acknowledgements

We gratefully acknowledge the Brahmal Vasudevan Institute for Sustainable Aviation at Imperial College London for funding this research and to the maintainers of the various scientific Python packages which `rojak` depends upon, such as `NumPy` [@harris2020array], `xarray` [@hoyer2017xarray], `Pandas` [@pandas], `GeoPandas` [@kelsey_jordahl_2020_3946761],  and `SciPy` [@virtanenSciPy10Fundamental2020].

# References

