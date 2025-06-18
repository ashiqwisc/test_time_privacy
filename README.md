# Test-Time Privacy Requires Going Beyond Certified Machine Unlearning

[Muhammad H. Ashiq](https://github.com/ashiqwisc)¹ · [Peter Triantafillou](https://warwick.ac.uk/fac/sci/dcs/people/peter_triantafillou/)² · [Hung Yun Tseng](https://openreview.net/profile?id=~Hung_Yun_Tseng1)¹ · [Grigoris G. Chrysos](https://grigorisg9gr.github.io/_pages/about/)¹  
¹University of Wisconsin-Madison · ²University of Warwick

[**Paper**](https://example.com) 

---

Unlearning is the predominant method for removing the influence of data in machine learning models. However, even after unlearning, models often continue to produce the same predictions on the unlearned data with high confidence. This persistent behavior can be exploited by adversaries using confident model predictions on incorrect or obsolete data to harm users. We call this threat model, which unlearning fails to protect against, _test-time privacy_. In particular, an adversary with full model access can bypass any naive defenses which ensure test-time privacy. To address this threat, we introduce an algorithm which perturbs model weights to induce maximal uncertainty on protected instances while preserving accuracy on the rest of the instances. Our core algorithm is based on finetuning with a Pareto optimal objective that explicitly balances test-time privacy against utility. We also provide a certifiable approximation algorithm which achieves (ε, δ)
 guarantees without convexity assumptions. We then prove a tight, non-vacuous bound that characterizes the privacy-utility tradeoff that our algorithms incur. Empirically, our method obtains a 4x privacy increase with a < 0.1% drop in performance on various image recognition benchmarks. Altogether, this framework provides a tool to guarantee additional protection to end users. 

The paper is currently under review. Code will be uploaded after the reviewers' decision is announced. 
