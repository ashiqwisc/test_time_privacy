# Inducing Uncertainty on Open-Weight Models for Test-Time Privacy in Image Recognition

[Muhammad H. Ashiq](https://github.com/ashiqwisc)¹ · [Peter Triantafillou](https://warwick.ac.uk/fac/sci/dcs/people/peter_triantafillou/)² · [Hung Yun Tseng](https://openreview.net/profile?id=~Hung_Yun_Tseng1)¹ · [Grigoris G. Chrysos](https://grigorisg9gr.github.io/_pages/about/)¹  
¹University of Wisconsin-Madison · ²University of Warwick

[**Paper**]([https://www.arxiv.org/abs/2509.11625]) 

---

A key concern for AI safety remains understudied in the machine learning (ML) literature: how can we ensure users of ML models do not leverage predictions on incorrect personal data to harm others? This is particularly pertinent given the rise of open-weight models, where simply masking model outputs does not suffice to prevent adversaries from recovering harmful predictions. To address this threat, which we call *test-time privacy*, we induce maximal uncertainty on protected instances while preserving accuracy on all other instances. Our proposed algorithm uses a Pareto optimal objective that explicitly balances test-time privacy against utility. We also provide a certifiable approximation algorithm which achieves $(\varepsilon, \delta)$ guarantees without convexity assumptions. We then prove a tight bound that characterizes the privacy-utility tradeoff that our algorithms incur. Empirically, our method obtains at least 3x stronger uncertainty than pretraining with marginal drops in accuracy on various image recognition benchmarks. Altogether, this framework provides a tool to guarantee additional protection to end users.

The paper was accepted as a long paper in the NeurIPS'25 workshop on Regulatable ML.
