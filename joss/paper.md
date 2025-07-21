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
date: XX June 2025
bibliography: paper.bib
---

# Summary

<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.

- [x] Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?
- [x] Explain the research applications of the software -->

<!-- Introduction to the problem/context it exists in -->
Aviation turbulence is atmospheric turbulence occurring at length scales large enough (approximately 100m to 1km) to affect an aircraft [@sharmanNatureAviationTurbulence2016]. According to the National Transport Safety Board (NTSB), turbulence experienced whilst onboard an aircraft was the leading cause of accidents from 2009 to 2018 [@ntsbPreventingTurbulenceRelatedInjuries2021].
Clear air turbulence (CAT) is a form of aviation turbulence which cannot be detected by the onboard weather radar. Thus, pilots are unable to preemptively avoid such regions.
In order to mitigate this safety risk, CAT diagnostics are used to forecast turbulent regions such that pilots are able to tactically avoid them.

<!-- Description of the software -->
`rojak` is a parallelised python library and command-line tool for using meteorological data to forecast CAT and evaluating the effectiveness of CAT diagnostics against turbulence observations.
Currently, it supports,

1. Computing turbulence diagnostics on meteorological data from European Centre for Medium-Range Weather Forecasts's (ECMWF) ERA5 reanalysis on pressure levels [@hersbach2023era5]. Moreover, it is easily extendable to support other types of meteorological data.
2. Retrieving and processing turbulence observations from Aircraft Meteorological Data Relay (AMDAR) data archived at National Oceanic and Atmospheric Administration (NOAA)[@NCEPMeteorologicalAssimilation] and AMDAR data collected via the Met Office MetDB system [@ukmoAmdar]
3. Computing 27 different turbulence diagnostics, such as the three-dimensional frontogenesis equation [@bluesteinSynopticdynamicMeteorologyMidlatitudes1993], turbulence index 1 and 2 [@ellrodObjectiveClearAirTurbulence1992], negative vorticity advection [@sharmanIntegratedApproachMid2006], and Brown's Richardson tendency equation [@brownNewIndicesLocate1973].
4. Converting turbulence diagnostic values into the eddy dissipation rate (EDR) --- the International Civil Aviation Organization's (ICAO) official metric for reporting turbulence [@icaoAnnex3]

These features not only allow users to perform operational forecasting of CAT but also to interrogate the intensification in frequency and severity of CAT due to climate change [@williamsIncreasedLightModerate2017; @storerGlobalResponseClearAir2017; @kimGlobalResponseUpperlevel2023], such as by analysing the climatological distribution of the probability of encountering turbulence at different severities (e.g. light turbulence or moderate-or-greater turbulence) for each turbulence diagnostic.
These applications involve high-volume datasets, ranging from tens to hundreds of gigabytes, necessitating the use of parallelisation to preserve computational tractability and efficiency, while substantially reducing execution time. As such, `rojak` leverages `Dask` to process larger-than-memory data and to run in a distributed manner [@pydataDask].

<!-- Research applications of the software -->
<!-- In particular, `rojak` opens up the following avenues of research,

1. Exploring whether CAT operational forecasts and our understanding on the effect of climate change on CAT can be improved. This could be achieved by evaluating turbulence diagnostics against observational data to uncover key drivers behind a given incidence of CAT. This could then be used to select the appropriate CAT diagnostic which captures the underlying phenomena. This targeted approach would narrow the uncertainties in the prediction.
2. Assessing the correlation to other atmospheric features (or phenomena) of interest, such as the likelihood of contrail formation. In the safety-focused aviation industry, it is vital to ensuring the climate-optimised flight trajectories --- which minimise the formation of contrails --- do not result in an increased risk due to turbulence.
3. Utilising turbulence diagnostics to inform the design of next-generation aircraft and gust load alleviation systems. As the aviation industry moves toward its net-zero goal, future aircraft are expected to be lighter and more aerodynamically efficient. This in turn makes them more susceptible to turbulence. A physics-based model of the intensified CAT enables a more robust aircraft design and the development of effective gust load mitigation strategies. -->

The name of the package, `rojak`, is inspired by its wide range turbulence diagnostics and its applications. While _rojak_ refers to a type of salad, it is also a colloquial term in Malaysia and Singapore for an eclectic mix, reflecting the diverse functionality of the package.

# Statement of need

<!-- A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.

- [x] Does the paper have a section titled ‘Statement of need’ that,
  - [x] Clearly states what problems the software is designed to solve
    - This has sort of been addressed in the summary part of it? So, do I need to re-iterate it?
    - I've reiterated it at the start
  - [x] Who the target audience is
    - I think this is sort of addressed?? It's anyone that's interested in aviation turbulence
  - [x] Its relation to other work?
- [x] State of the field: Do the authors describe how this software compares to other commonly-used packages? -->

<!-- Operationally, the forecasting of CAT is vital to ensuring passenger. However, with anthropogenic climate change, it is increasingly important to understand the climatological distribution of CAT and how it will change under its effects. -->

<!-- Link to aforementioned problems it solves -->
Numerous studies have investigated the influence of the climate on CAT [e.g. @kimGlobalResponseUpperlevel2023; @williamsIncreasedLightModerate2017; @storerGlobalResponseClearAir2017] and the application of turbulence diagnostics for operational forecasting [e.g. @gillObjectiveVerificationWorld2014; @sharmanIntegratedApproachMid2006; @sharmanPredictionEnergyDissipation2017; @gillEnsembleBasedTurbulence2014] in products like the Federal Aviation Authority's (FAA) Graphical Turbulence Guidance and the International Civil Aviation Organization's (ICAO) World Area Forecast System.
<!-- State of the field -->
However, to the best of the author's knowledge, none of these studies have made their code publicly available. This work presents the first open-source package for aviation turbulence analysis.
<!-- aiming to improve accessibility into this field for the wider research community. -->
Given the inherent complexity of CAT diagnostics and the variability in how these could be implemented, `rojak` serves as a first iteration of a standardised implementation of these CAT diagnostics, providing a basis for future enhancements and refinements by the broader research community.
Moreover, the parallelised nature of `rojak` and its architecture, which keeps it open to extensions, positions it as as an indispensable resource to bridging this gap.

![Probability of encountering light turbulence during the months December, January, February from 2018 to 2024 at 200 hPa for the three-dimensional frontogenesis (F3D) and turbulence index 1 (TI1) diagnostics \label{fig:probability_light_turbulence}](multi_diagnostic_f3d_ti1_on_200_light.png)

<!-- Relation to other works - Aviation Meteorology Literature -->
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

<!-- For instance, \ref{probability_light_turbulence} depicts the global climatological distribution the probability of encountering light turbulence for the boreal winter months (i.e., December, January and February) from 2018 to 2024 at 200 hPa based on the two turbulence diagnostic --- three-dimensional frontogenesis equation and turbulence index 1. This was computed using ERA5 data at 6-hourly intervals with three pressure levels (175 hPa, 200 hPa and 225 hPa) for the aforementioned time period. The thresholds to determine if a given diagnostics value is sufficiently large to indicate the presence of light turbulence was established using 6-hourly data obtained on the 1st and 15th of every month from 1980 to 1989 to find the values in the 97 to 99.1 percentiles.
This was computed through `rojak`'s implementation of the methodology described in @williamsCanClimateModel2022.
The full details and references of the are contained within the documentation of `rojak`. -->

<!-- Bridge to the need for observational data  -->
In the context of operational forecasting [e.g. methods detailed in @gillObjectiveVerificationWorld2014; @sharmanIntegratedApproachMid2006; and @pearsonPredictionEnergyDissipation2017], the comparison of turbulence diagnostics computed from meteorological data against observational data is a fundamental component. The statistical nature of using an ensemble of turbulence diagnostics which has an optimal balance of a low false positive and false negative rate mandates it.
The architecture of `rojak` enables it to seamlessly integrate various sources of meteorological and observational data, with their interactions managed through a central mediator.
<!-- The design of `rojak` abstracts interactions between various meteorological and observational data sources through a mediator, ensuring extensibility and modularity. -->
<!-- As such `rojak` has not only been architected to handle different source of meteorological data and observational data, but also abstracted for their interaction to be through a mediator. -->
<!-- Bridge to talking about CLI to retrieve data -->
`rojak` also contains a command-line interface tool which can be launched to perform a variety of aviation turbulence analyses and to retrieve the meteorological and observation data from the various data providers.

<!-- 1. Methodology for evaluating the presence of turbulence of a given severity comes from literature, e.g. thresholds for probabilities and EDR
2. Evaluating the efficacy of a given turbulence diagnostics against observational data

Talk about the architecture?? Architected to handle different source of meteorological data and observational data. Through the command line, it is also possible to retrieve the data from the various providers. Also has the ability to perform geo-spatial analysis??? Mention the other core features. Provides both a library and CLI interface -->

<!-- Relation to other works - MetPy -->
In terms of the calculations performed upon the meteorological data, `MetPy` [@mayMetPyPythonPackage2016] has the greatest similarity to `rojak`. However, it does not natively support `Dask` [@manserSupportDaskArrays]. Given the size of the datasets to be processed, this presented a significant issue.
Moreover, `MeyPy` does not implement the calculations required by the turbulence diagnostics.

<!-- # General requirements

- [ ] Quality of writing: Is the paper well written (i.e., it does not require editing for structure, language, or writing quality)? -->

# Acknowledgements

We gratefully acknowledge the Brahmal Vasudevan Institute for Sustainable Aviation at Imperial College London for funding this research and to the maintainers of the various scientific Python packages which `rojak` depends upon, such as `NumPy` [@harris2020array], `xarray` [@hoyer2017xarray], `Pandas` [@pandas], `GeoPandas` [@kelsey_jordahl_2020_3946761],  and `SciPy` [@virtanenSciPy10Fundamental2020].

# References

<!-- - [ ] Is the list of references complete, and is everything cited appropriately that should be cited (e.g., papers, datasets, software)? Do references in the text use the proper citation syntax? -->
