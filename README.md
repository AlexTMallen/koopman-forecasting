# Deep Probabilistic Koopman (DPK): Long-term time-series forecasting under periodic uncertainties
**Stable, long-term, probabilistic forecasts with calibrated uncertainty
measures.** Our Koopman-theoretic approach enables powerful forecasts of
a variety of quasi-periodic phenomena from  electricity demand to neural
activity.

**For a more user-friendly DPK framework, see this
[repository](https://github.com/AlexTMallen/dpk).**

## Dependencies
The following are dependencies of the DPK framework:
- torch
- numpy
- scipy

While these are used for the experiments in the paper:
- pandas
- sklearn
- matplotlib
- jupyter
- allensdk (optional—only install this if you wish to work with the
  neuroscience data)

All of these are available through `pip`.

## Results
All results can be found in `figures.ipynb` with instructions on how to
replicate them. In a retrospective comparison, our model outperforms all 177 competing teams in Global
Energy Forecasting Competition 2017 at the qualifying task. We also show significant improvements over a NASA air quality forecast model.

## Training a DPK model
1. Load your time-series data into memory as a time-by-n numpy array.  
   (If the samples are not uniform over time, you will also need to
   provide a 1D array of these times).
2. Input the frequencies your data exhibits. This is usually as easy as
   24 hours, 1 week, and 1 year, but sometimes the frequencies must be
   found by DPK by solving a global optimization problem or via the FFT.
3. Choose a model object from `model_obs.py` or write your own. We
   recommend starting out with `SkewNLLwithTime`, which assumes your
   data is drawn from a time-varying skew-normal distribution at every
   point in time. "withTime" indicates that this model object allows for
   non-periodic trends.
4. Call fit!

```
x = np.sin(np.linspace(0, 1000 * np.pi, 10000)).reshape(-1, 1)  # for example
periods = [20,]  # 20 idxs is a period
model_obj = model_objs.SkewNLLwithTime(x_dim=x.shape[1], num_freqs=[len(periods),] * 3)

k = koopman_probabilistic.KoopmanProb(model_obj, device='cpu', num_fourier_modes=len(periods))
k.find_fourier_omegas(x, hard_code=periods)
k.fit(x, iterations=25, verbose=True)
```

## Forecasting
1. Call predict! This returns the time-varying parameters of the
   distribution defined by your model object. To obtain a point forecast
   with uncertainty, simply call `model_obj.mean(params)` and
   `model_obj.std(params)`. This step should be instantaneous since DPK
   does not require time-stepping for predictions.

```
params = k.predict(T=15000)
loc_hat, scale_hat, alpha_hat = params  # time-varying parameters of skew-normal distribution
x_hat = model_obj.mean(params)
std_hat = model_obj.std(params)  # uncertainty over time
```

A similar but slightly more sophisticated example can be found in
`example.py`.
