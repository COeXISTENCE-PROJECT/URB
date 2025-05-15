Apart from RL immplementations we provide baseline algorithms to compare with, can be used with ```baselines.py``` and model options consist:
* **Baselines included in URB**
    * ```aon``` model which deterministically picks the shortest free-flow route regardless of the congestion,
    * ```random``` model which is fully undeterministic,
* **Additionally, available from `RouteRL`**
    * ```gawron``` (i.e. base human learning model) model is based on: [Gawron (1998)](<https://kups.ub.uni-koeln.de/9257/>), the model iteratively shifts the cost expectations towards the received reward.
