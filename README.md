# Retrospective on the Capstone competition

The working files and retrospective of the competition and approaches.

## The progression of the strategy

The initial intention was to somehow make use of the existing algorithms such
as [Heteroscedastic Evolutionary Bayesian Optimisation](https://github.com/huawei-noah/HEBO/tree/master/HEBO)  to start
with the competition. However, there was an example, [example implementation](example_student.ipynb) introduced in one
of the office hours for the first function with `GaussianProcessRegressor` as the surrogate model with `UCB` as
the acquisition function. This example also included a plot of `UCB` values for a specific grid.
The example also included a 3D scatter plot of the whole data set for the first function. This helped in
understanding the context of the competition and also helped visualise the data, which also helped in other functions
too.

The plot of `UCB` was based on the `Grid Search` and the next point was suggested by means of a maximum point of `UCB`
There was also an example of how `Random Search` can be used to get the next set of points to reach the maximum
point of the function. What was of interest from these examples was the plot of the datapoints to show where their outputs
were concentrated. Added to it were the instructions about *Function 1*, which mentioned that there are only 2
sources of radiation and that the radiations could only be detected when the points are very close to them.
This meant that it was logical to `exploit` the area where we got indications on the plot as to where the concentration
was. The search area of the grid search was reduced around [0.65,0.65] and grid search was conducted using `UCB`
as acquisition function again. Since *Function 2* also had 2 input variables, the *plotting* route was
tempting, although the instructions mentioned that there were a lot of local optima. The same strategy however,
could not be used for *Function 3* onwards because the number of input variables were more than 2.

Since `UCB` in general could be used as an acquisition function in Bayesian Optimisation
it was decided to generalise the `UCB` based example to the other functions too.
Hence, [example_student_multi_dimensional.ipynb](example_student_multi_dimensional.ipynb) was developed
to produce the entries for the whole set of functions. The output that this notebook produced indicated that
the maximum point with the limited number of attempts wasn't going to possible. The reason was that a few of the
repeated attempts produced 0.000000 or 1.000000 continuously.

Therefore, an existing black box function optimisation algorithm,
[HEBO](https://github.com/huawei-noah/HEBO/tree/master/HEBO) was explored the notebook
[hebo_observe_and_suggest.ipynb](hebo_observe_and_suggest.ipynb) was developed. It was evident from the outset that
this notebook could produce some meaningful entry points. The discomfort back then was that the functions used in this
notebook were not fully understood. To an extent the functions themselves were used as blackbox functions.
Because the weeks contained other intense materials, it was not possible to allocate enough resources
to analyse the methods used in the notebook. Information such as the acquisition functions and surrogate models were not
fully known by the time of the implementation of this notebook.

This prompted an effort to start working on another algorithm based on `Expected Improvement` acquisition function and
a notebook with EI as acquisition function, [multi_dimensional_with_ei.ipynb](multi_dimensional_with_ei.ipynb) was
written. This also produced some entries that received reasonable feedback for the higher order functions.
This was the approximate time when the first leaderboard was announced and the results at that stage proved that the
something was right in the approach. However, since there were no special treatments allocated for different functions
based on their said behaviour on the challenge, the approach was assumed to be somewhat sub-optimal.
Nevertheless, to see the outcome of these notebooks for a few weeks, the entries for the next few weeks were
produced using these two (`hebo_observe_and_suggest` and `multi_dimensional_with_ei`) notebooks.
The entries for *Function 1* and *Function 2* changed intermittently based on the scatter plots obtained using the
`example_student` notebook.

This approach was, to an extent blindly, adopted until the second set of initial data,
[initial_data2](initial_data2) was introduced and later another approach based on
[BoTorch](https://botorch.org/docs/introduction) was also taught in an office hour.
The difference that piqued the interest on this approach was the step where the state was
updated with the latest observation before the next points were proposed. This then prompted an attempt to specialise
the code for individual functions based on their initial information in the competition.
As such different acquisition functions were allocated different functions based on their suitability for each function.

Based on some research, __Thomson Sampling__ turned out to be a suitable acquisition for data with many local optima
and for data with noise. This also appeared to be suitable for high dimensional data due to its performance on local
exploration. Hence, _TS_ was used as the acquisition function for Functions 1,3, 4 and 8. __Expected Improvement__
on the other hand was seemingly suitable for exploitation, faster convergence in unimodal setups
while allowing some degree of exploration. This meant that this was the most suitable acquisition function
for Functions 2, 5 and 7. The clue on _Function 6_ was however a little mystic and based on some brief internet/chatgpt
search, __Knowledge Gradient__ appeared to suite this type of function for its better accuracy,
balance of exploration and exploitation and for multi objective optimisation.
The resultant script was [turbo_final.py](turbo_final.py).

The information about _Function 3_ in the introduction also mentioned that in this drug discovery problem, one of the
input variables had no impact on the output of the experiment. This information, along with the fact that it was a 3
variable function prompted an attempt to standardise the inputs and to filter out non-impacting variable towards
the end of the competition. To do this, `StandardScaler` and `Lasso` classes from `sklearn` were used to reduce this
set of 3 variables into 2. This change was incorporated in the `turbo_final` script.

Finally, while nearing the end of the competition there were about 52 records of inputs and outputs, that prompted
an an approach to experiment to fit the data to any known functions to come up with their mean and standard deviation.
It was thought that if such an attempt was possible the maximum of the function with 2 variables would be where the
input variables equalled mean values. Proceeding with this approach, `curve_fit` function from `scipy` was used to fit
the data to a 2D Gaussian function and for _Function 1_ the mean value where the expected peak must lie, turned out as
something that was reasonably close to the initial exploration area of [0.65, 0.65]. Another observation that helped
this approach with the first function was that the function appeared to be symmetric with respect to its input
variables. This was spotted when a couple of entries with interchanged values were fed to the function and the result
turned out to be equal. See [first_fn_final.ipynb](first_fn_final.ipynb)

The same approach, bar the equality of the two variables, was attempted to _Funciton 2_ but the set of entries did not
appear to be on the right direction based on the feedback received. And as for _Function 3_, since it was suspected
that only the 2nd and 3rd variables influenced the output of the function, the same approach was attempted to fit
the values of the 2nd and 3rd variables with the output values to a 2D Gaussian function but `curve_fit` function
with this set of data did not converge. An attempt to plot variables 2 and 3 and against the output also did not
produce any meaningful suggestion. See [second_fn_final.ipynb](second_function_final.ipynb)

## Summary

In summary the initial entries were produced using
[example_student_multi_dimension](example_student_multi_dimensional.ipynb) notebook. Then outputs from
[hebo_observe_and_suggest](hebo_observe_and_suggest.ipynb) and from
[multi_dimensional_with_ei](multi_dimensional_with_ei.ipynb) were interchangeably used.
Finally, [turbo_final](turbo_final.py) was used for most of the entries with some specialisations for Functions 1 and 2.

Although, this combined approach produced inputs that ranked the 8th overall in the competition, given another chance
and time there would be attempts made to start specialising the approaches per individual functions and HEBO based
approach would be given more weight since it was highly suitable for cases where there is a lot of local optima and for
cases of high dimensionality.
