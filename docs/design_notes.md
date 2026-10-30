# Design notes: what the published frameworks say makes a good JITAI

This is a short summary of **published** frameworks, with quotations where the
wording matters. It contains no clinical claims of its own. When this page says
"the authors", it means the paper cited in that section.

## 1. The six components of a JITAI (Nahum-Shani et al., 2018)

Nahum-Shani et al. describe a JITAI as an intervention design that adapts the
provision of support (type, timing, intensity) "over time to an individual's
changing status and contexts", with the goal of delivering support "at the moment
and in the context that the person needs it most and is most likely to be
receptive" (quoting their ref. 1). They state that "a JITAI
includes 6 key elements: a distal outcome, proximal outcomes, decision points,
intervention options, tailoring variables, and decision rules." Their glossary
gives these definitions (quoted):

| Component | Definition (Nahum-Shani et al., 2018) | Where it shows up in `jitailab` |
|---|---|---|
| Distal outcome | "The ultimate goal the intervention is intended to achieve" | not modelled (MRTs target proximal effects) |
| Proximal outcomes | "The short-term goals the intervention is intended to achieve"; they "can be mediators ... and/or intermediate measures of the distal outcome" | `Y` (the outcome after each decision point) |
| Decision points | "A decision point is a time at which an intervention decision is made" | the index `t` |
| Intervention options | "an array of possible treatments or actions that might be employed at any given decision point" | binary `A_t` (nudge vs. no nudge) |
| Tailoring variables | "information concerning the individual that is used to decide when ... to provide an intervention and which intervention to provide" | context `S_t`, availability `I_t` |
| Decision rules | link "the intervention options and tailoring variables in a systematic way" | the randomization policy / a future deterministic rule |

Two further concepts from the same paper matter directly for design:

* **State of vulnerability / opportunity**: "a period of susceptibility to
  negative health outcomes (vulnerability) or to positive health behavior changes
  (opportunity)". Decision points should be placed where such states can occur and
  change.
* **Receptivity**: "the individual's transient ability and/or willingness to
  receive, process, and utilize just-in-time support". The authors note the
  underlying assumption that support delivered when a person is not receptive is
  less useful.

On the timing of decision points, the authors observe that the interval between
them should match how fast the tailoring variable is expected to change in a
meaningful way, and that this choice "can have a dramatic impact on the ability of
the adaptation to achieve its goal".

On burden, the authors discuss intervention fatigue and list, among its
mechanisms, cognitive overload and **habituation** ("objective decline in
physiological and/or behavioral response to an intervention over repeated
exposures"). As a design strategy they mention explicitly including a **"provide
nothing" intervention option** in the decision rules. In `jitailab`, habituation
and burden are simulation knobs (`eta`, `zeta`), not empirical estimates.

## 2. Why micro-randomize (Klasnja et al., 2015)

Klasnja et al. introduce the micro-randomized trial (MRT) as an experimental
design for developing JITAIs. Points relevant to this toolkit:

* **What it is.** Intervention options are randomized at each decision point, for
  each participant, so a study lasting weeks can randomize each person hundreds
  of times; with several components randomized concurrently, the authors call
  micro-randomization "a form of a sequential factorial design".
* **What it answers.** The abstract states that MRTs "enable causal modeling of
  proximal effects of the randomized intervention components and assessment of
  time-varying moderation of those effects". The paper also discusses lagged
  effects.
* **Proximal outcomes** are "the effects that the delivery of a particular
  intervention component is intended to have", often seen as mediators of the
  distal outcome. A first goal of an MRT is to check whether components have
  their intended proximal effects.
* **Availability.** When an intervention "is not feasible or appropriate at a
  decision point, we consider the individual unavailable"; randomization happens
  only when the participant is available, and the authors stress that keeping
  track of availability matters for effect estimation. In `jitailab` this is the
  indicator `I_t`, and all estimators condition on `I_t = 1`.
* **A worked example (HeartSteps).** The paper describes activity suggestions
  randomized at five time points per day, with a 40% probability chosen "to
  reduce participant burden", over 42 days, i.e. 210 decision points per person.
  `scripts/study_power.py` borrows this 42 x 5, p = 0.4 layout; the availability
  rate 0.7 used there is our assumption, not a number from the paper.

## 3. Analysis and sizing (the methods implemented here)

* **Causal excursion effects and WCLS**: Boruvka et al. (2018) define the
  time-varying (moderated) proximal effect and the weighted and centered
  least-squares estimator implemented in `jitailab.wcls`.
* **Sample size**: Liao et al. (2016) derive the sample-size formula implemented
  in `jitailab.power`; Klasnja et al. (2015) note that MRT primary analyses can be
  powered with regression-based methods and refer to it.
* **A practitioner-oriented review** of MRT design and analysis choices is Qian
  et al. (2022).
* **Why naive mixed models can mislead** when covariates are affected by past
  treatment: Qian, Klasnja & Murphy (2020).
* **Adaptive randomization in practice**: Liao et al. (2020) describe a
  Thompson-sampling-based personalization of HeartSteps. The `ThompsonPolicy`
  in this repo is a simplified per-participant teaching device, not that algorithm.

## References

All DOIs were resolved on Crossref on 2026-10-30.

* Boruvka, A., Almirall, D., Witkiewitz, K., & Murphy, S. A. (2018). Assessing time-varying causal effect moderation in mobile health. *Journal of the American Statistical Association, 113*(523), 1112–1121. https://doi.org/10.1080/01621459.2017.1305274
* Klasnja, P., Hekler, E. B., Shiffman, S., Boruvka, A., Almirall, D., Tewari, A., & Murphy, S. A. (2015). Microrandomized trials: An experimental design for developing just-in-time adaptive interventions. *Health Psychology, 34*(Suppl), 1220–1228. https://doi.org/10.1037/hea0000305
* Liao, P., Klasnja, P., Tewari, A., & Murphy, S. A. (2016). Sample size calculations for micro-randomized trials in mHealth. *Statistics in Medicine, 35*(12), 1944–1971. https://doi.org/10.1002/sim.6847
* Liao, P., Greenewald, K., Klasnja, P., & Murphy, S. (2020). Personalized HeartSteps: A reinforcement learning algorithm for optimizing physical activity. *Proc. ACM IMWUT, 4*(1), 1–22. https://doi.org/10.1145/3381007
* Nahum-Shani, I., Smith, S. N., Spring, B. J., Collins, L. M., Witkiewitz, K., Tewari, A., & Murphy, S. A. (2018). Just-in-time adaptive interventions (JITAIs) in mobile health: Key components and design principles for ongoing health behavior support. *Annals of Behavioral Medicine, 52*(6), 446–462. https://doi.org/10.1007/s12160-016-9830-8
* Qian, T., Klasnja, P., & Murphy, S. A. (2020). Linear mixed models with endogenous covariates: Modeling sequential treatment effects with application to a mobile health study. *Statistical Science, 35*(3), 375–390. https://doi.org/10.1214/19-STS720
* Qian, T., Walton, A. E., Collins, L. M., Klasnja, P., Lanza, S. T., Nahum-Shani, I., Rabbi, M., Russell, M. A., Walton, M. A., Yoo, H., & Murphy, S. A. (2022). The microrandomized trial for developing digital interventions: Experimental design and data analysis considerations. *Psychological Methods, 27*(5), 874–894. https://doi.org/10.1037/met0000283
