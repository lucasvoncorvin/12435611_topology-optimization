## Evaluation

My evaluations are mainly based on comparing my results with the results from [CITE PAPER]. I will focus on key metrics such as compliance, volume fraction, and material density to assess the quality of the predicted structures. The paper reports 5 models of different sizes. My models are relatively small due to available computational resources and time constraints. I will therefore compare my results primarily with the smallest model "Tiny", which had roughly 5 million parameters and is the most comparable in terms of size.

### Compliance

The point of strucutural topology optimization as it is done in this project is to maximize the stiffness of a structure while sticking to a given material budget, represented by the volume fraction of material used. The main factor for usability is the structural integratiy, therefore, to evaluate the model prediction for the structure, I will compute the compliance (the inverse of stiffness) of the predicted structure and compare it to the compliance of the ground truth structure. 

The compliance is computed in [CITE FUNCTION] using the same functionality that was used to generate the dataset using the SIMP method. (I will describe this in detail in the final report)

**Target**

The authors of the paper [CITE PAPER] state an achieved average compliance error of 4.75% and a median compliance error of 1.55% for their "Tiny" model (~5M parameters), which serves as a benchmark for comparison. The goal is not to beat this value, since this might be unrealistic for the scope of this project. However, I will aim get reasonably close to these values with my smaller models. I decided on the following values:

| Metric | Target Value |
|--------|--------------|
| Average Compliance Error | <= 5% |
| Median Compliance Error | <=2% |


### Shape

One potential application of a topology optimization model like the one I develop is to quickly get an idea of the rough shape of the structure, for example to evaluate available or required design space in the early stages of the design process, or to decide on suitable manufacturing methods for a given shape.

I will use this to define a minimum viable model. I want my model to predict shapes that overlap with the ground truth shapes at least to 90% on average.


### Volume

Furthermore, in practice, topology optimized mechanical components are often manufactured via casting (i.e. molten material is poured into a mold and allowed to solidify). One of the main cost drivers is actually the material volumen. For this reason, a good model should adhere to the specified material budget (i.e. volume fraction [0, 1]). I will therefore also report the volume fraction of the predicted structure, however I will not set strict requirements on it. It will be used a metric for reporting how well the model respects the material budget. Unreasonably large values will be investigated and the model adjusted accordingly.


### Material Density

Another thing to consider is that the result of our topology optimization is typically a binary material distribution. A 1 in a pixel indicates the presence of material, while a 0 its absence. A inherent property of the method we used to create the ground truth structures (so called SIMP method) is that it allows continuous values between 0 and 1. However, they have no physical meaning, since material is either present or absent in reality. The SIMP method handles this by penalizing intermediate values (Thats the P in SIMP, which stands for Penalization), this way we this approach tends to converge to almost binary values. We want our model to produce predictions that are also close to binary as well, however we cannot expect it to be more binary than the ground truth itself. Evaluate how much the predicted structure deviates from being strictly binary compared to the ground truth.

Similar to the volume fraction I will report this metric, however it will be a purely informative measure and not a strict requirement.

**Material Density in other metrics**

I will compute the compliance on the predicted structures rounded to binary values as well as with continuous densities. The continous density compliance will be the one that is used for comparison with the target compliance. The binary compliance will be reported as an additional metric to understand how well the predicted structure performs when strictly interpreted as a binary material distribution.

For the shape overlap, I will compute the intersection over union (IoU) between the predicted and target topologies after rounding them to binary values. This will give an indication of how well the predicted shape matches the ground truth shape, with a higher value indicating better overlap.

### Note on Error Metrics and Acceptable Values

Deriving acceptable error metrics for topology optimization results is very challenging. At the present form the topology optimization used in this project (2D, SIMP) serves at best as a rough idea of how a mechanical component should look like. The practicality of the result of the ground truth structures is therefore very limited. In practice topology optimized structure must fullfill a lot of requirements like:

- manufacturing constraints (very hard to satisfy)
- CAD compatibility and digital representation 
- geometric constraints (e.g., minimum feature size, connectivity, symmetry)
- Maximum stresses
- safety factors

So topology optimization require a thorough post processing and/or more sophisticated methods to be practically useful. All of the above could also be considered for evaluating the predicted structures, depending on the intended application. Given the variety of requirements and constraints, there are just as many ways to evaluate the quality of a predicted structure.

Furthermore, the usefulness of the predicted structures varies widely depending on the actual application and priorities of the designer. For some applications a 5% compliance error might be acceptable, while for others even a 1% error could be critical. A 2% volumen fraction error can cause significant cost in terms of excessive material usage when a million components are being produced. It is however probably more than acceptable for a custom made 3D printed component.

I did therefore refrain from trying to derive acceptable error metrics for the predicted structures from practical considerations, since the ground truth structures themselves are hardly of any practical use in this form. They are more of a tool for studying and exploring. And even for this purpose each user must decide for themselves what constitutes an acceptable result. Hence, I based my evaluation primarily on the comparison with other deep learning models.


### Summary

| Metric | Description | Value |
|--------|-------------|-------|
| Compliance Error (Continuous) | Error in compliance for the predicted structure with continuous densities | Average <= 5%, median <= 2% |
| Compliance Error (Binary) | Error in compliance for the predicted structure with binary densities |  |
| Shape Overlap | Overlap between the predicted and target topologies | Average >= 90% |
| Volume Error | Error in the volume of the predicted structure compared to the target structure |  |
| Density Error | Error indicating how far the predicted densities deviate from valid binary values (0 or 1) compared to the deviation of the target densities | |



