Axiomatic Attribution for Deep Networks
Mukund Sundararajan * 1 Ankur Taly* 1 Qiqi Yan* 1
Abstract
We study the problem of attributing the pre-
diction of a deep network to its input features,
a problem previously studied by several other
works. We identify two fundamental axioms—
Sensitivity and Implementation Invariance that
attribution methods ought to satisfy. We show
that they are not satisﬁed by most known attri-
bution methods, which we consider to be a fun-
damental weakness of those methods. We use
the axioms to guide the design of a new attri-
bution method called Integrated Gradients. Our
method requires no modiﬁcation to the original
network and is extremely simple to implement;
it just needs a few calls to the standard gradi-
ent operator. We apply this method to a couple
of image models, a couple of text models and a
chemistry model, demonstrating its ability to de-
bug networks, to extract rules from a network,
and to enable users to engage with models better.
1. Motivation and Summary of Results
We study the problem ofattributing the prediction of a deep
network to its input features.
Deﬁnition 1. Formally, suppose we have a function F :
Rn → [0, 1] that represents a deep network, and an in-
putx = (x1,...,x n)∈ Rn. An attribution of the predic-
tion at input x relative to a baseline input x′ is a vector
AF (x,x′) = (a1,...,a n)∈ Rn whereai is the contribu-
tion ofxi to the predictionF (x).
For instance, in an object recognition network, an attribu-
tion method could tell us which pixels of the image were
responsible for a certain label being picked (see Figure 2).
The attribution problem was previously studied by vari-
ous papers (Baehrens et al., 2010; Simonyan et al., 2013;
*Equal contribution 1Google Inc., Mountain View,
USA. Correspondence to: Mukund Sundararajan
<mukunds@google.com>, Ankur Taly<ataly@google.com>.
Proceedings of the 34th International Conference on Machine
Learning, Sydney, Australia, PMLR 70, 2017. Copyright 2017
by the author(s).
Shrikumar et al., 2016; Binder et al., 2016; Springenberg
et al., 2014).
The intention of these works is to understand the input-
output behavior of the deep network, which gives us the
ability to improve it. Such understandability is critical to
all computer programs, including machine learning mod-
els. There are also other applications of attribution. They
could be used within a product driven by machine learn-
ing to provide a rationale for the recommendation. For in-
stance, a deep network that predicts a condition based on
imaging could help inform the doctor of the part of the im-
age that resulted in the recommendation. This could help
the doctor understand the strengths and weaknesses of a
model and compensate for it. We give such an example in
Section 6.2. Attributions could also be used by developers
in an exploratory sense. For instance, we could use a deep
network to extract insights that could be then used in a rule-
based system. In Section 6.3, we give such an example.
A signiﬁcant challenge in designing an attribution tech-
nique is that they are hard to evaluate empirically. As we
discuss in Section 4, it is hard to tease apart errors that stem
from the misbehavior of the model versus the misbehavior
of the attribution method. To compensate for this short-
coming, we take an axiomatic approach. In Section 2 we
identify two axioms that every attribution method must sat-
isfy. Unfortunately most previous methods do not satisfy
one of these two axioms. In Section 3, we use the axioms
to identify a new method, called integrated gradients.
Unlike previously proposed methods, integrated gradients
do not need any instrumentation of the network, and can
be computed easily using a few calls to the gradient opera-
tion, allowing even novice practitioners to easily apply the
technique.
In Section 6, we demonstrate the ease of applicability over
several deep networks, including two images networks, two
text processing networks, and a chemistry network. These
applications demonstrate the use of our technique in either
improving our understanding of the network, performing
debugging, performing rule extraction, or aiding an end
user in understanding the network’s prediction.
Remark 1. Let us brieﬂy examine the need for the base-
line in the deﬁnition of the attribution problem. A common
way for humans to perform attribution relies on counter-
arXiv:1703.01365v2  [cs.LG]  13 Jun 2017

Axiomatic Attribution for Deep Networks
factual intuition. When we assign blame to a certain cause
we implicitly consider the absence of the cause as a base-
line for comparing outcomes. In a deep network, we model
the absence using a single baseline input. For most deep
networks, a natural baseline exists in the input space where
the prediction is neutral. For instance, in object recognition
networks, it is the black image. The need for a baseline has
also been pointed out by prior work on attribution (Shriku-
mar et al., 2016; Binder et al., 2016).
2. Two Fundamental Axioms
We now discuss two axioms (desirable characteristics) for
attribution methods. We ﬁnd that other feature attribution
methods in literature break at least one of the two axioms.
These methods include DeepLift (Shrikumar et al., 2016;
2017), Layer-wise relevance propagation (LRP) (Binder
et al., 2016), Deconvolutional networks (Zeiler & Fergus,
2014), and Guided back-propagation (Springenberg et al.,
2014). As we will see in Section 3, these axioms will also
guide the design of our method.
Gradients. For linear models, ML practitioners regularly
inspect the products of the model coefﬁcients and the fea-
ture values in order to debug predictions. Gradients (of the
output with respect to the input) is a natural analog of the
model coefﬁcients for a deep network, and therefore the
product of the gradient and feature values is a reasonable
starting point for an attribution method (Baehrens et al.,
2010; Simonyan et al., 2013); see the third column of Fig-
ure 2 for examples. The problem with gradients is that
they break sensitivity, a property that all attribution meth-
ods should satisfy.
2.1. Axiom: Sensitivity(a)
An attribution method satisﬁes Sensitivity(a) if for every
input and baseline that differ in one feature but have differ-
ent predictions then the differing feature should be given
a non-zero attribution. (Later in the paper, we will have a
part (b) to this deﬁnition.)
Gradients violate Sensitivity(a): For a concrete example,
consider a one variable, one ReLU network, f(x) = 1 −
ReLU(1−x). Suppose the baseline isx = 0 and the input is
x = 2. The function changes from 0 to 1, but becausef be-
comes ﬂat atx = 1, the gradient method gives attribution of
0 tox. Intuitively, gradients break Sensitivity because the
prediction function may ﬂatten at the input and thus have
zero gradient despite the function value at the input being
different from that at the baseline. This phenomenon has
been reported in previous work (Shrikumar et al., 2016).
Practically, the lack of sensitivity causes gradients to focus
on irrelevant features (see the “ﬁreboat” example in Fig-
ure 2).
Other back-propagation based approaches. A second
set of approaches involve back-propagating the ﬁnal pre-
diction score through each layer of the network down to the
individual features. These include DeepLift, Layer-wise
relevance propagation (LRP), Deconvolutional networks
(DeConvNets), and Guided back-propagation. These meth-
ods differ in the speciﬁc backpropagation logic for various
activation functions (e.g., ReLU, MaxPool, etc.).
Unfortunately, Deconvolution networks (DeConvNets),
and Guided back-propagation violate Sensitivity(a). This
is because these methods back-propogate through a ReLU
node only if the ReLU is turned on at the input. This makes
the method similar to gradients, in that, the attribution is
zero for features with zero gradient at the input despite a
non-zero gradient at the baseline. We defer the speciﬁc
counterexamples to Appendix B.
Methods like DeepLift and LRP tackle the Sensitivity issue
by employing a baseline, and in some sense try to compute
“discrete gradients” instead of (instantaeneous) gradients at
the input. (The two methods differ in the speciﬁcs of how
they compute the discrete gradient). But the idea is that a
large, discrete step will avoid ﬂat regions, avoiding a break-
age of sensitivity. Unfortunately, these methods violate a
different requirement on attribution methods.
2.2. Axiom: Implementation Invariance
Two networks are functionally equivalent if their outputs
are equal for all inputs, despite having very different imple-
mentations. Attribution methods should satisfy Implemen-
tation Invariance, i.e., the attributions are always identical
for two functionally equivalent networks. To motivate this,
notice that attribution can be colloquially deﬁned as assign-
ing the blame (or credit) for the output to the input features.
Such a deﬁnition does not refer to implementation details.
We now discuss intuition for why DeepLift and LRP break
Implementation Invariance; a concrete example is provided
in Appendix B.
First, notice that gradients are invariant to implementation.
In fact, the chain-rule for gradients ∂f
∂g = ∂f
∂h· ∂h
∂g is essen-
tially about implementation invariance. To see this, think
ofg andf as the input and output of a system, andh being
some implementation detail of the system. The gradient of
outputf to inputg can be computed either directly by ∂f
∂g ,
ignoring the intermediate function h (implementation de-
tail), or by invoking the chain rule via h. This is exactly
how backpropagation works.
Methods like LRP and DeepLift replace gradients with dis-
crete gradients and still use a modiﬁed form of backpropa-
gation to compose discrete gradients into attributions. Un-

Axiomatic Attribution for Deep Networks
fortunately, the chain rule does not hold for discrete gra-
dients in general. Formally f (x1)−f (x0)
g(x1)−g(x0) ̸= f (x1)−f (x0)
h(x1)−h(x0)·
h(x1)−h(x0)
g(x1)−g(x0) , and therefore these methods fail to satisfy im-
plementation invariance.
If an attribution method fails to satisfy Implementation In-
variance, the attributions are potentially sensitive to unim-
portant aspects of the models. For instance, if the network
architecture has more degrees of freedom than needed to
represent a function then there may be two sets of values
for the network parameters that lead to the same function.
The training procedure can converge at either set of values
depending on the initializtion or for other reasons, but the
underlying network function would remain the same. It is
undesirable that attributions differ for such reasons.
3. Our Method: Integrated Gradients
We are now ready to describe our technique. Intuitively,
our technique combines the Implementation Invariance of
Gradients along with the Sensitivity of techniques like LRP
or DeepLift.
Formally, suppose we have a functionF : Rn→ [0, 1] that
represents a deep network. Speciﬁcally, letx∈ Rn be the
input at hand, andx′∈ Rn be the baseline input. For image
networks, the baseline could be the black image, while for
text models it could be the zero embedding vector.
We consider the straightline path (in Rn) from the baseline
x′ to the input x, and compute the gradients at all points
along the path. Integrated gradients are obtained by cu-
mulating these gradients. Speciﬁcally, integrated gradients
are deﬁned as the path intergral of the gradients along the
straightline path from the baselinex′ to the inputx.
The integrated gradient along theith dimension for an input
x and baselinex′ is deﬁned as follows. Here, ∂F (x)
∂xi
is the
gradient ofF (x) along theith dimension.
IntegratedGradsi(x) ::= (xi −x′
i) ×
∫ 1
α=0
∂F (x′+α×(x−x′))
∂xi
dα
(1)
Axiom: Completeness. Integrated gradients satisfy an
axiom called completeness that the attributions add up to
the difference between the output of F at the input x and
the baselinex′. This axiom is identiﬁed as being desirable
by Deeplift and LRP. It is a sanity check that the attribu-
tion method is somewhat comprehensive in its accounting,
a property that is clearly desirable if the networks score is
used in a numeric sense, and not just to pick the top la-
bel, for e.g., a model estimating insurance premiums from
credit features of individuals.
This is formalized by the proposition below, which instanti-
ates the fundamental theorem of calculus for path integrals.
Proposition 1. If F : Rn → R is differentiable almost
everywhere 1 then
Σn
i=1IntegratedGradsi(x) =F (x)−F (x′)
For most deep networks, it is possible to choose a base-
line such that the prediction at the baseline is near zero
(F (x′)≈ 0). (For image models, the black image base-
line indeed satisﬁes this property.) In such cases, there is
an intepretation of the resulting attributions that ignores the
baseline and amounts to distributing the output to the indi-
vidual input features.
Remark 2. Integrated gradients satisﬁes Sensivity(a) be-
cause Completeness implies Sensivity(a) and is thus a
strengthening of the Sensitivity(a) axiom. This is because
Sensitivity(a) refers to a case where the baseline and the
input differ only in one variable, for which Completeness
asserts that the difference in the two output values is equal
to the attribution to this variable. Attributions generated
by integrated gradients satisfy Implementation Invariance
since they are based only on the gradients of the function
represented by the network.
4. Uniqueness of Integrated Gradients
Prior literature has relied on empirically evaluating the at-
tribution technique. For instance, in the context of an object
recognition task, (Samek et al., 2015) suggests that we se-
lect the topk pixels by attribution and randomly vary their
intensities and then measure the drop in score. If the at-
tribution method is good, then the drop in score should be
large. However, the images resulting from pixel perturba-
tion could be unnatural, and it could be that the scores drop
simply because the network has never seen anything like it
in training. (This is less of a concern with linear or logis-
tic models where the simplicity of the model ensures that
ablating a feature does not cause strange interactions.)
A different evaluation technique considers images with
human-drawn bounding boxes around objects, and com-
putes the percentage of pixel attribution inside the box.
While for most objects, one would expect the pixels located
on the object to be most important for the prediction, in
some cases the context in which the object occurs may also
contribute to the prediction. The cabbage butterﬂy image
from Figure 2 is a good example of this where the pixels
on the leaf are also surfaced by the integrated gradients.
Roughly, we found that every empirical evaluation tech-
nique we could think of could not differentiate between ar-
1Formally, this means the function F is continuous every-
where and the partial derivative ofF along each input dimension
satisﬁes Lebesgue’s integrability condition, i.e., the set of discon-
tinuous points has measure zero. Deep networks built out of Sig-
moids, ReLUs, and pooling operators satisfy this condition.

Axiomatic Attribution for Deep Networks
r1,r2
s1,s2
P1
P2 P3
Figure 1. Three paths between an a baseline (r1,r 2) and an input
(s1,s 2). Each path corresponds to a different attribution method.
The pathP2 corresponds to the path used by integrated gradients.
tifacts that stem from perturbing the data, a misbehaving
model, and a misbehaving attribution method. This was
why we turned to an axiomatic approach in designing a
good attribution method (Section 2). While our method
satisﬁes Sensitivity and Implementation Invariance, it cer-
tainly isn’t the unique method to do so.
We now justify the selection of the integrated gradients
method in two steps. First, we identify a class of meth-
ods called Path methods that generalize integrated gradi-
ents. We discuss that path methods are the only methods
to satisfy certain desirable axioms. Second, we argue why
integrated gradients is somehow canonical among the dif-
ferent path methods.
4.1. Path Methods
Integrated gradients aggregate the gradients along the in-
puts that fall on the straightline between the baseline and
the input. There are many other (non-straightline) paths
that monotonically interpolate between the two points, and
each such path will yield a different attribution method. For
instance, consider the simple case when the input is two di-
mensional. Figure 1 has examples of three paths, each of
which corresponds to a different attribution method.
Formally, letγ = (γ1,...,γ n) : [0, 1]→ Rn be a smooth
function specifying a path in Rn from the baselinex′ to the
inputx, i.e.,γ(0) =x′ andγ(1) =x.
Given a path function γ, path integrated gradients are ob-
tained by integrating the gradients along the path γ(α) for
α∈ [0, 1]. Formally, path integrated gradients along the
ith dimension for an inputx is deﬁned as follows.
PathIntegratedGradsγ
i (x) ::=
∫ 1
α=0
∂F (γ(α))
∂γi(α)
∂γi(α)
∂α dα
(2)
where ∂F (x)
∂xi
is the gradient of F along the ith dimension
atx.
Attribution methods based on path integrated gradients are
collectively known as path methods. Notice that integrated
gradients is a path method for the straightline path speciﬁed
γ(α) =x′ +α× (x−x′) forα∈ [0, 1].
Remark 3. All path methods satisfy Implementation In-
variance. This follows from the fact that they are deﬁned
using the underlying gradients, which do not depend on the
implementation. They also satisfy Completeness (the proof
is similar to that of Proposition 1) and Sensitvity(a) which
is implied by Completeness (see Remark 2).
More interestingly, path methods are the only methods
that satisfy certain desirable axioms. (For formal deﬁni-
tions of the axioms and proof of Proposition 2, see Fried-
man (Friedman, 2004).)
Axiom: Sensitivity(b). (called Dummy in (Friedman,
2004)) If the function implemented by the deep network
does not depend (mathematically) on some variable, then
the attribution to that variable is always zero.
This is a natural complement to the deﬁnition of Sensitiv-
ity(a) from Section 2. This deﬁnition captures desired in-
sensitivity of the attributions.
Axiom: Linearity. Suppose that we linearly composed
two deep networks modeled by the functions f1 andf2 to
form a third network that models the functiona×f1+b×f2,
i.e., a linear combination of the two networks. Then we’d
like the attributions fora×f1 +b×f2 to be the weighted
sum of the attributions for f1 andf2 with weightsa andb
respectively. Intuitively, we would like the attributions to
preserve any linearity within the network.
Proposition 2. (Theorem 1 (Friedman, 2004)) Path meth-
ods are the only attribution methods that always satisfy
Implementation Invariance, Sensitivity(b), Linearity, and
Completeness.
Remark 4. We note that these path integrated gradients
have been used within the cost-sharing literature in eco-
nomics where the function models the cost of a project as
a function of the demands of various participants, and the
attributions correspond to cost-shares. Integrated gradi-
ents correspond to a cost-sharing method called Aumann-
Shapley (Aumann & Shapley, 1974). Proposition 2 holds
for our attribution problem because mathematically the
cost-sharing problem corresponds to the attribution prob-
lem with the benchmark ﬁxed at the zero vector. (Imple-
mentation Invariance is implicit in the cost-sharing litera-
ture as the cost functions are considered directly in their
mathematical form.)
4.2. Integrated Gradients is Symmetry-Preserving
In this section, we formalize why the straightline path cho-
sen by integrated gradients is canonical. First, observe that
it is the simplest path that one can deﬁne mathematically.

Axiomatic Attribution for Deep Networks
Second, a natural property for attribution methods is to pre-
serve symmetry, in the following sense.
Symmetry-Preserving. Two input variables are symmet-
ric w.r.t. a function if swapping them does not change the
function. For instance, x and y are symmetric w.r.t. F if
and only ifF (x,y ) =F (y,x ) for all values ofx andy. An
attribution method is symmetry preserving, if for all inputs
that have identical values for symmetric variables and base-
lines that have identical values for symmetric variables, the
symmetric variables receive identical attributions.
E.g., consider the logistic model Sigmoid(x1 +x2 +... ).
x1 andx2 are symmetric variables for this model. For an
input where x1 = x2 = 1 (say) and baseline where x1 =
x2 = 0 (say), a symmetry preserving method must offer
identical attributions tox1 andx2.
It seems natural to ask for symmetry-preserving attribution
methods because if two variables play the exact same role
in the network (i.e., they are symmetric and have the same
values in the baseline and the input) then they ought to re-
ceive the same attrbiution.
Theorem 1. Integrated gradients is the unique path
method that is symmetry-preserving.
The proof is provided in Appendix A.
Remark 5. If we allow averaging over the attributions
from multiple paths, then are other methods that satisfy all
the axioms in Theorem 1. In particular, there is the method
by Shapley-Shubik (Shapley & Shubik, 1971) from the cost
sharing literature, and used by (Lundberg & Lee, 2016;
Datta et al., 2016) to compute feature attributions (though
they were not studying deep networks). In this method, the
attribution is the average of those from n! extremal paths;
heren is the number of features. Here each such path con-
siders an ordering of the input features, and sequentially
changes the input feature from its value at the baseline to
its value at the input. This method yields attributions that
are different from integrated gradients. If the function of
interest ismin(x1,x 2), the baseline is x1 = x2 = 0, and
the input is x1 = 1 , x2 = 3 , then integrated gradients
attributes the change in the function value entirely to the
critical variablex1, whereas Shapley-Shubik assigns attri-
butions of 1/2 each; it seems somewhat subjective to prefer
one result over the other.
We also envision other issues with applying Shapley-Shubik
to deep networks: It is computationally expensive; in an
object recognition network that takes an 100X100 image
as input, n is 10000, and n! is a gigantic number. Even
if one samples few paths randomly, evaluating the attribu-
tions for a single path takes n calls to the deep network.
In contrast, integrated gradients is able to operate with 20
to 300 calls. Further, the Shapley-Shubik computation visit
inputs that are combinations of the input and the baseline.
It is possible that some of these combinations are very dif-
ferent from anything seen during training. We speculate
that this could lead to attribution artifacts.
5. Applying Integrated Gradients
Selecting a Benchmark. A key step in applying integrated
gradients is to select a good baseline. We recommend that
developers check that the baseline has a near-zero score—
as discussed in Section 3, this allows us to interpret the
attributions as a function of the input. But there is more to
a good baseline: For instance, for an object recogntion net-
work it is possible to create an adversarial example that has
a zero score for a given input label (say elephant), by apply-
ing a tiny, carefully-designed perturbation to an image with
a very different label (say microscope) (cf. (Goodfellow
et al., 2015)). The attributions can then include undesirable
artifacts of this adversarially constructed baseline. So we
would additionally like the baseline to convey a complete
absence of signal, so that the features that are apparent from
the attributions are properties only of the input, and not of
the baseline. For instance, in an object recognition net-
work, a black image signiﬁes the absence of objects. The
black image isn’t unique in this sense—an image consisting
of noise has the same property. However, using black as a
baseline may result in cleaner visualizations of “edge” fea-
tures. For text based networks, we have found that the all-
zero input embedding vector is a good baseline. The action
of training causes unimportant words tend to have small
norms, and so, in the limit, unimportance corresponds to
the all-zero baseline. Notice that the black image corre-
sponds to a valid input to an object recognition network,
and is also intuitively what we humans would consider ab-
sence of signal. In contrast, the all-zero input vector for a
text network does not correspond to a valid input; it never-
theless works for the mathematical reason described above.
Computing Integrated Gradients. The integral of inte-
grated gradients can be efﬁciently approximated via a sum-
mation. We simply sum the gradients at points occurring at
sufﬁciently small intervals along the straightline path from
the baselinex′ to the inputx.
IntegratedGradsapprox
i (x) ::=
(xi−x′
i)× Σm
k=1
∂F (x′+k
m×(x−x′)))
∂xi
× 1
m
(3)
Here m is the number of steps in the Riemman approxi-
mation of the integral. Notice that the approximation sim-
ply involves computing the gradient in a for loop which
should be straightforward and efﬁcient in most deep learn-
ing frameworks. For instance, in TensorFlow, it amounts
to calling tf.gradients in a loop over the set of in-
puts (i.e., x′ + k
m× (x−x′) for k = 1,...,m ), which

Axiomatic Attribution for Deep Networks
could also be batched. In practice, we ﬁnd that somewhere
between 20 and 300 steps are enough to approximate the
integral (within 5%); we recommend that developers check
that the attributions approximately adds up to the differ-
ence beween the score at the input and that at the baseline
(cf. Proposition 1), and if not increase the step-sizem.
6. Applications
The integrated gradients technique is applicable to a variety
of deep networks. Here, we apply it to two image models,
two natural language models, and a chemistry model.
6.1. An Object Recognition Network
We study feature attribution in an object recognition net-
work built using the GoogleNet architecture (Szegedy
et al., 2014) and trained over the ImageNet object recog-
nition dataset (Russakovsky et al., 2015). We use the inte-
grated gradients method to study pixel importance in pre-
dictions made by this network. The gradients are computed
for the output of the highest-scoring class with respect to
pixel of the input image. The baseline input is the black
image, i.e., all pixel intensities are zero.
Integrated gradients can be visualized by aggregating them
along the color channel and scaling the pixels in the ac-
tual image by them. Figure 2 shows visualizations for a
bunch of images2. For comparison, it also presents the cor-
responding visualization obtained from the product of the
image with the gradients at the actual image. Notice that
integrated gradients are better at reﬂecting distinctive fea-
tures of the input image.
6.2. Diabetic Retinopathy Prediction
Diabetic retinopathy (DR) is a complication of the diabetes
that affects the eyes. Recently, a deep network (Gulshan
et al., 2016) has been proposed to predict the severity grade
for DR in retinal fundus images. The model has good pre-
dictive accuracy on various validation datasets.
We use integrated gradients to study feature importance for
this network; like in the object recognition case, the base-
line is the black image. Feature importance explanations
are important for this network as retina specialists may use
it to build trust in the network’s predictions, decide the
grade for borderline cases, and obtain insights for further
testing and screening.
Figure 3 shows a visualization of integrated gradients for a
retinal fundus image. The visualization method is a bit dif-
ferent from that used in Figure 2. We aggregate integrated
gradients along the color channel and overlay them on the
2More examples can be found at https://github.com/
ankurtaly/Attributions
Figure 2. Comparing integrated gradients with gradients at
the image. Left-to-right: original input image, label and softmax
score for the highest scoring class, visualization of integrated gra-
dients, visualization of gradients*image. Notice that the visual-
izations obtained from integrated gradients are better at reﬂecting
distinctive features of the image.
actual image in gray scale with positive attribtutions along
the green channel and negative attributions along the red
channel. Notice that integrated gradients are localized to a
few pixels that seem to be lesions in the retina. The inte-
rior of the lesions receive a negative attribution while the
periphery receives a positive attribution indicating that the
network focusses on the boundary of the lesion.
Figure 3. Attribution for Diabetic Retinopathy grade predic-
tion from a retinal fundus image. The original image is show
on the left, and the attributions (overlayed on the original image
in gray scaee) is shown on the right. On the original image we an-
notate lesions visible to a human, and conﬁrm that the attributions
indeed point to them.

Axiomatic Attribution for Deep Networks
6.3. Question Classiﬁcation
Automatically answering natural language questions (over
semi-structured data) is an important problem in artiﬁcial
intelligence (AI). A common approach is to semantically
parse the question to its logical form (Liang, 2016) using
a set of human-authored grammar rules. An alternative ap-
proach is to machine learn an end-to-end model provided
there is enough training data. An interesting question is
whether one could peek inside machine learnt models to de-
rive new rules. We explore this direction for a sub-problem
of semantic parsing, called question classiﬁcation, using
the method of integrated gradients.
The goal of question classiﬁcation is to identify the type of
answer it is seeking. For instance, is the quesiton seek-
ing a yes/no answer, or is it seeking a date? Rules for
solving this problem look for trigger phrases in the ques-
tion, for e.g., a “when” in the beginning indicates a date
seeking question. We train a model for question classiﬁca-
tion using the the text categorization architecture proposed
by (Kim, 2014) over the WikiTableQuestions dataset (Pasu-
pat & Liang, 2015). We use integrated gradients to attribute
predictions down to the question terms in order to identify
new trigger phrases for answer type. The baseline input is
the all zero embedding vector.
Figure 4 lists a few questions with constituent terms high-
lighted based on their attribution. Notice that the attri-
butions largely agree with commonly used rules, for e.g.,
“how many” indicates a numeric seeking question. In ad-
dition, attributions help identify novel question classiﬁca-
tion rules, for e.g., questions containing “total number” are
seeking numeric answers. Attributions also point out unde-
sirable correlations, for e.g., “charles” is used as trigger for
a yes/no question.
Figure 4. Attributions from question classiﬁcation model.
Term color indicates attribution strength—Red is positive, Blue is
negative, and Gray is neutral (zero). The predicted class is speci-
ﬁed in square brackets.
6.4. Neural Machine Translation
We applied our technique to a complex, LSTM-based Neu-
ral Machine Translation System (Wu et al., 2016). We
attribute the output probability of every output token (in
form of wordpieces) to the input tokens. Such attributions
“align” the output sentence with the input sentence. For
baseline, we zero out the embeddings of all tokens except
the start and end markers. Figure 5 shows an example of
such an attribution-based alignments. We observed that the
results make intuitive sense. E.g. “und” is mostly attributed
to “and”, and “morgen” is mostly attributed to “morning”.
We use 100− 1000 steps (cf. Section 5) in the integrated
gradient approximation; we need this because the network
is highly nonlinear.
Figure 5. Attributions from a language translation model. In-
put in English: “good morning ladies and gentlemen”. Output in
German: “Guten Morgen Damen und Herren”. Both input and
output are tokenized into word pieces, where a word piece pre-
ﬁxed by underscore indicates that it should be the preﬁx of a word.
6.5. Chemistry Models
We apply integrated gradients to a network performing
Ligand-Based Virtual Screening which is the problem of
predicting whether an input molecule is active against a
certain target (e.g., protein or enzyme). In particular, we
consider a network based on the molecular graph convolu-
tion architecture proposed by (Kearnes et al., 2016).
The network requires an input molecule to be encoded by
hand as a set of atom and atom-pair features describing the
molecule as an undirected graph. Atoms are featurized us-
ing a one-hot encoding specifying the atom type (e.g., C, O,
S, etc.), and atom-pairs are featurized by specifying either
the type of bond (e.g., single, double, triple, etc.) between
the atoms, or the graph distance between them. The base-
line input is obtained zeroing out the feature vectors for
atom and atom-pairs.
We visualize integrated gradients as heatmaps over the the
atom and atom-pair features with the heatmap intensity de-
picting the strength of the contribution. Figure 6 shows
the visualization for a speciﬁc molecule. Since integrated
gradients add up to the ﬁnal prediction score (see Proposi-
tion 1), the magnitudes can be use for accounting the con-
tributions of each feature. For instance, for the molecule in
the ﬁgure, atom-pairs that have a bond between them cu-
mulatively contribute to 46% of the prediction score, while
all other pairs cumulatively contribute to only−3%.

Axiomatic Attribution for Deep Networks
Figure 6. Attribution for a molecule under the W2N2 net-
work (Kearnes et al., 2016) . The molecules is active on task
PCBA-58432.
Identifying Degenerate Features. We now discuss how
attributions helped us spot an anomaly in the W1N2 ar-
chitecture in (Kearnes et al., 2016). On applying the in-
tegrated gradients method to this network, we found that
several atoms in the same molecule received identical at-
tribution despite being bonded to different atoms. This is
surprising as one would expect two atoms with different
neighborhoods to be treated differently by the network.
On investigating the problem further, in the network archi-
tecture, the atoms and atom-pair features were not fully
convolved. This caused all atoms that have the same atom
type, and same number of bonds of each type to contribute
identically to the network.
7. Other Related work
We already covered closely related work on attribution in
Section 2. We mention other related work. Over the last
few years, there has been a vast amount work on demysti-
fying the inner workings of deep networks. Most of this
work has been on networks trained on computer vision
tasks, and deals with understanding what a speciﬁc neu-
ron computes (Erhan et al., 2009; Le, 2013) and interpret-
ing the representations captured by neurons during a pre-
diction (Mahendran & Vedaldi, 2015; Dosovitskiy & Brox,
2015; Yosinski et al., 2015). In contrast, we focus on un-
derstanding the network’s behavior on a speciﬁc input in
terms of the base level input features. Our technique quan-
tiﬁes the importance of each feature in the prediction.
One approach to the attribution problem proposed ﬁrst
by (Ribeiro et al., 2016a;b), is to locally approximate the
behavior of the network in the vicinity of the input being
explained with a simpler, more interpretable model. An
appealing aspect of this approach is that it is completely
agnostic to the implementation of the network and satisﬁes
implemenation invariance. However, this approach does
not guarantee sensitivity. There is no guarantee that the
local region explored escapes the “ﬂat” section of the pre-
diction function in the sense of Section 2. The other issue
is that the method is expensive to implement for networks
with “dense” input like image networks as one needs to ex-
plore a local region of size proportional to the number of
pixels and train a model for this space. In contrast, our
technique works with a few calls to the gradient operation.
Attention mechanisms (Bahdanau et al., 2014) have gained
popularity recently. One may think that attention could be
used a proxy for attributions, but this has issues. For in-
stance, in a LSTM that also employs attention, there are
many ways for an input token to inﬂuence an output to-
ken: the memory cell, the recurrent state, and “attention”.
Focussing only an attention ignores the other modes of in-
ﬂuence and results in an incomplete picture.
8. Conclusion
The primary contribution of this paper is a method called
integrated gradients that attributes the prediction of a deep
network to its inputs. It can be implemented using a few
calls to the gradients operator, can be applied to a variety
of deep networks, and has a strong theoretical justiﬁcation.
A secondary contribution of this paper is to clarify desir-
able features of an attribution method using an axiomatic
framework inspired by cost-sharing literature from eco-
nomics. Without the axiomatic approach it is hard to tell
whether the attribution method is affected by data arti-
facts, network’s artifacts or artifacts of the method. The
axiomatic approach rules out artifacts of the last type.
While our and other works have made some progress on
understanding the relative importance of input features in
a deep network, we have not addressed the interactions
between the input features or the logic employed by the
network. So there remain many unanswered questions in
terms of debugging the I/O behavior of a deep network.
ACKNOWLEDGMENTS
We would like to thank Samy Bengio, Kedar Dhamdhere,
Scott Lundberg, Amir Najmi, Kevin McCurley, Patrick Ri-
ley, Christian Szegedy, Diane Tang for their feedback. We
would like to thank Daniel Smilkov and Federico Allocati
for identifying bugs in our descriptions. We would like to
thank our anonymous reviewers for identifying bugs, and
their suggestions to improve presentation.
References
Aumann, R. J. and Shapley, L. S. Values of Non-Atomic
Games. Princeton University Press, Princeton, NJ, 1974.
Baehrens, David, Schroeter, Timon, Harmeling, Stefan,
Kawanabe, Motoaki, Hansen, Katja, and M ¨uller, Klaus-

Axiomatic Attribution for Deep Networks
Robert. How to explain individual classiﬁcation deci-
sions. Journal of Machine Learning Research, pp. 1803–
1831, 2010.
Bahdanau, Dzmitry, Cho, Kyunghyun, and Bengio,
Yoshua. Neural machine translation by jointly learning
to align and translate. CoRR, abs/1409.0473, 2014. URL
http://arxiv.org/abs/1409.0473.
Binder, Alexander, Montavon, Gr ´egoire, Bach, Sebastian,
M¨uller, Klaus-Robert, and Samek, Wojciech. Layer-
wise relevance propagation for neural networks with lo-
cal renormalization layers. CoRR, 2016.
Datta, A., Sen, S., and Zick, Y . Algorithmic transparency
via quantitative input inﬂuence: Theory and experiments
with learning systems. In 2016 IEEE Symposium on Se-
curity and Privacy (SP), pp. 598–617, 2016.
Dosovitskiy, Alexey and Brox, Thomas. Inverting visual
representations with convolutional networks, 2015.
Erhan, Dumitru, Bengio, Yoshua, Courville, Aaron, and
Vincent, Pascal. Visualizing higher-layer features of a
deep network. Technical Report 1341, University of
Montreal, 2009.
Friedman, Eric J. Paths and consistency in additive cost
sharing. International Journal of Game Theory , 32(4):
501–518, 2004.
Goodfellow, Ian, Shlens, Jonathon, and Szegedy, Chris-
tian. Explaining and harnessing adversarial exam-
ples. In International Conference on Learning Repre-
sentations, 2015. URL http://arxiv.org/abs/
1412.6572.
Gulshan, Varun, Peng, Lily, Coram, Marc, and et al. Devel-
opment and validation of a deep learning algorithm for
detection of diabetic retinopathy in retinal fundus pho-
tographs. JAMA, 316(22):2402–2410, 2016.
Kearnes, Steven, McCloskey, Kevin, Berndl, Marc, Pande,
Vijay, and Riley, Patrick. Molecular graph convolutions:
moving beyond ﬁngerprints.Journal of Computer-Aided
Molecular Design, pp. 595–608, 2016.
Kim, Yoon. Convolutional neural networks for sentence
classiﬁcation. In ACL, 2014.
Le, Quoc V . Building high-level features using large scale
unsupervised learning. In International Conference on
Acoustics, Speech, and Signal Processing (ICASSP), pp.
8595–8598, 2013.
Liang, Percy. Learning executable semantic parsers for nat-
ural language understanding. Commun. ACM, 59(9):68–
76, 2016.
Lundberg, Scott and Lee, Su-In. An unexpected unity
among methods for interpreting model predictions.
CoRR, abs/1611.07478, 2016. URL http://arxiv.
org/abs/1611.07478.
Mahendran, Aravindh and Vedaldi, Andrea. Understand-
ing deep image representations by inverting them. In
Conference on Computer Vision and Pattern Recognition
(CVPR), pp. 5188–5196, 2015.
Pasupat, Panupong and Liang, Percy. Compositional se-
mantic parsing on semi-structured tables. In ACL, 2015.
Ribeiro, Marco T´ulio, Singh, Sameer, and Guestrin, Carlos.
”why should I trust you?”: Explaining the predictions of
any classiﬁer. In22nd ACM International Conference on
Knowledge Discovery and Data Mining, pp. 1135–1144.
ACM, 2016a.
Ribeiro, Marco T ´ulio, Singh, Sameer, and Guestrin, Car-
los. Model-agnostic interpretability of machine learning.
CoRR, 2016b.
Russakovsky, Olga, Deng, Jia, Su, Hao, Krause, Jonathan,
Satheesh, Sanjeev, Ma, Sean, Huang, Zhiheng, Karpa-
thy, Andrej, Khosla, Aditya, Bernstein, Michael, Berg,
Alexander C., and Fei-Fei, Li. ImageNet Large Scale
Visual Recognition Challenge. International Journal of
Computer Vision (IJCV), pp. 211–252, 2015.
Samek, Wojciech, Binder, Alexander, Montavon, Gr´egoire,
Bach, Sebastian, and M ¨uller, Klaus-Robert. Evaluat-
ing the visualization of what a deep neural network has
learned. CoRR, 2015.
Shapley, Lloyd S. and Shubik, Martin. The assignment
game : the core. International Journal of Game Theory,
1(1):111–130, 1971. URL http://dx.doi.org/
10.1007/BF01753437.
Shrikumar, Avanti, Greenside, Peyton, Shcherbina, Anna,
and Kundaje, Anshul. Not just a black box: Learning
important features through propagating activation differ-
ences. CoRR, 2016.
Shrikumar, Avanti, Greenside, Peyton, and Kundaje, An-
shul. Learning important features through propagating
activation differences. CoRR, abs/1704.02685, 2017.
URL http://arxiv.org/abs/1704.02685.
Simonyan, Karen, Vedaldi, Andrea, and Zisserman, An-
drew. Deep inside convolutional networks: Visualising
image classiﬁcation models and saliency maps. CoRR,
2013.
Springenberg, Jost Tobias, Dosovitskiy, Alexey, Brox,
Thomas, and Riedmiller, Martin A. Striving for sim-
plicity: The all convolutional net. CoRR, 2014.

Axiomatic Attribution for Deep Networks
Szegedy, Christian, Liu, Wei, Jia, Yangqing, Sermanet,
Pierre, Reed, Scott E., Anguelov, Dragomir, Erhan, Du-
mitru, Vanhoucke, Vincent, and Rabinovich, Andrew.
Going deeper with convolutions. CoRR, 2014.
Wu, Yonghui, Schuster, Mike, Chen, Zhifeng, Le, Quoc V .,
Norouzi, Mohammad, Macherey, Wolfgang, Krikun,
Maxim, Cao, Yuan, Gao, Qin, Macherey, Klaus,
Klingner, Jeff, Shah, Apurva, Johnson, Melvin, Liu,
Xiaobing, Kaiser, Lukasz, Gouws, Stephan, Kato,
Yoshikiyo, Kudo, Taku, Kazawa, Hideto, Stevens, Keith,
Kurian, George, Patil, Nishant, Wang, Wei, Young, Cliff,
Smith, Jason, Riesa, Jason, Rudnick, Alex, Vinyals,
Oriol, Corrado, Greg, Hughes, Macduff, and Dean, Jef-
frey. Google’s neural machine translation system: Bridg-
ing the gap between human and machine translation.
CoRR, abs/1609.08144, 2016. URL http://arxiv.
org/abs/1609.08144.
Yosinski, Jason, Clune, Jeff, Nguyen, Anh Mai, Fuchs,
Thomas, and Lipson, Hod. Understanding neural net-
works through deep visualization. CoRR, 2015.
Zeiler, Matthew D. and Fergus, Rob. Visualizing and un-
derstanding convolutional networks. In ECCV, pp. 818–
833, 2014.
A. Proof of Theorem 1
Proof. Consider a non-straightline path γ : [0, 1]→ Rn
from baseline to input. W.l.o.g., there exists t0 ∈ [0, 1]
such that for two dimensions i,j , γi(t0) > γj(t0). Let
(t1,t 2) be the maximum real open interval containing t0
such that γi(t) > γj(t) for all t in (t1,t 2), and let a =
γi(t1) =γj(t1), andb =γi(t2) =γj(t2). Deﬁne function
f : x∈ [0, 1]n→ R as 0 if min(xi,xj)≤ a, as (b−a)2
if max(xi,xj)≥ b, and as (xi−a)(xj−a) otherwise.
Next we compute the attributions of f atx =⟨1,..., 1⟩n
with baseline x′ =⟨0,..., 0⟩n. Note that xi and xj are
symmetric, and should get identical attributions. For t /∈
[t1,t 2], the function is a constant, and the attribution of f
is zero to all variables, while for t∈ (t1,t 2), the integrand
of attribution of f isγj(t)−a toxi, and γi(t)−a toxj,
where the latter is always strictly larger by our choice of
the interval. Integrating, it follows that xj gets a larger
attribution thanxi, contradiction.
B. Attribution Counter-Examples
We show that the methods DeepLift and Layer-wise rel-
evance propagation (LRP) break the implementation in-
variance axiom, and the Deconvolution and Guided back-
propagation methods break the sensitivity axiom.
Figure 7 provides an example of two equivalent networks
Networkf (x1,x 2)
Attributions atx1 = 3,x 2 = 1
Integrated gradients x1 = 1.5, x2 = −0.5
DeepLift x1 = 1.5, x2 = −0.5
LRP x1 = 1.5, x2 = −0.5
Networkg(x1,x 2)
Attributions atx1 = 3,x 2 = 1
Integrated gradients x1 = 1.5, x2 = −0.5
DeepLift x1 = 2, x2 = −1
LRP x1 = 2, x2 = −1
Figure 7. Attributions for two functionally equivalent net-
works. The ﬁgure shows attributions for two functionally equiva-
lent networksf (x1,x 2) andg(x1,x 2) at the inputx1 = 3, x2 =
1 using integrated gradients, DeepLift (Shrikumar et al., 2016),
and Layer-wise relevance propagation (LRP) (Binder et al.,
2016). The reference input for Integrated gradients and DeepLift
is x1 = 0, x2 = 0 . All methods except integrated gradients
provide different attributions for the two networks.
f(x1,x 2) andg(x1,x 2) for which DeepLift and LRP yield
different attributions.
First, observe that the networks f and g are of the
form f(x1,x 2) = ReLU(h(x1,x 2)) and f(x1,x 2) =
ReLU(k(x1,x 2))3, where
h(x1,x 2) = ReLU(x1)− 1− ReLU(x2)
k(x1,x 2) = ReLU(x1− 1)− ReLU(x2)
Note that h and k are not equivalent. They have differ-
ent values whenever x1 < 1. But f andg are equivalent.
To prove this, suppose for contradiction that f and g are
different for some x1,x 2. Then it must be the case that
ReLU(x1)− 1̸= ReLU(x1− 1). This happens only when
x1 < 1, which implies thatf(x1,x 2) =g(x1,x 2) = 0.
Now we leverage the above example to show that Deconvo-
lution and Guided back-propagation break sensitivity. Con-
sider the network f(x1,x 2) from Figure 7. For a ﬁxed
value of x1 greater than 1, the output decreases linearly
asx2 increases from 0 tox1− 1. Yet, for all inputs, De-
convolutional networks and Guided back-propagation re-
sults in zero attribution for x2. This happens because for
all inputs the back-propagated signal received at the node
ReLU(x2) is negative and is therefore not back-propagated
through the ReLU operation (per the rules of deconvolu-
tion and guided back-propagation; see (Springenberg et al.,
2014) for details). As a result, the feature x2 receives zero
3 ReLU(x) is deﬁned asmax(x, 0).

Axiomatic Attribution for Deep Networks
attribution despite the network’s output being sensitive to
it.