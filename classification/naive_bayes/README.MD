The very best explanation for Naive Bayes can be sought from <a href="http://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification">this</a> post on StackOverflow. The 2nd answer from the top says it best. I'll copy paste it here for convenience.

I realize that this is an old question, with an established answer. The reason I'm posting is that is the accepted answer has many elements of k-NN (k-nearest neighbors), a different algorithm.

Both k-NN and NaiveBayes are classification algorithms. Conceptually, k-NN uses the idea of "nearness" to classify new entities. In k-NN 'nearness' is modeled with ideas such as Euclidean Distance or Cosine Distance. By contrast, in NaiveBayes, the concept of 'probability' is used to classify new entities.

Since the question is about Naive Bayes, here's how I'd describe the ideas and steps to someone. I'll try to do it with as few equations and in plain English as much as possible.

First, Conditional Probability & Bayes' Rule

Before someone can understand and appreciate the nuances of Naive Bayes', they need to know a couple of related concepts first, namely, the idea of Conditional Probability, and Bayes' Rule. (If you are familiar with these concepts, skip to the section titled Getting to Naive Bayes')

Conditional Probability in plain English: What is the probability that something will happen, given that something else has already happened.

Let's say that there is some Outcome O. And some Evidence E. From the way these probabilities are defined: The Probability of having both the Outcome O and Evidence E is: (Probability of O occurring) multiplied by the (Prob of E given that O happened)

One Example to understand Conditional Probability:

Let say we have a collection of US Senators. Senators could be Democrats or Republicans. They are also either male or female.

If we select one senator completely randomly, what is the probability that this person is a female Democrat? Conditional Probability can help us answer that.

Probability of (Democrat and Female Senator)= Prob(Senator is Democrat) multiplied by Conditional Probability of Being Female given that they are a Democrat.

  P(Democrat & Female) = P(Democrat) * P(Female | Democrat)
  We could compute the exact same thing, the reverse way:

    P(Democrat & Female) = P(Female) * P(Democrat | Female)
    Understanding Bayes Rule

    Conceptually, this is a way to go from P(Evidence| Known Outcome) to P(Outcome|Known Evidence). Often, we know how frequently some particular evidence is observed, given a known outcome. We have to use this known fact to compute the reverse, to compute the chance of that outcome happening, given the evidence.

    P(Outcome given that we know some Evidence) = P(Evidence given that we know the Outcome) times Prob(Outcome), scaled by the P(Evidence)

    The classic example to understand Bayes' Rule:

    Probability of Disease D given Test-positive =

                   Prob(Test is positive|Disease) * P(Disease)
		        _______________________________________________________________
			     (scaled by) Prob(Testing Positive, with or without the disease)
			     Now, all this was just preamble, to get to Naive Bayes.

			     Getting to Naive Bayes'

			     So far, we have talked only about one piece of evidence. In reality, we have to predict an outcome given multiple evidence. In that case, the math gets very complicated. To get around that complication, one approach is to 'uncouple' multiple pieces of evidence, and to treat each of piece of evidence as independent. This approach is why this is called naive Bayes.

			     P(Outcome|Multiple Evidence) =
			     P(Evidence1|Outcome) * P(Evidence2|outcome) * ... * P(EvidenceN|outcome) * P(Outcome)
			     scaled by P(Multiple Evidence)
			     Many people choose to remember this as:

			                           P(Likelihood of Evidence) * Prior prob of outcome
						   P(outcome|evidence) = _________________________________________________
						                                            P(Evidence)
											    Notice a few things about this equation:

											    If the Prob(evidence|outcome) is 1, then we are just multiplying by 1.
											    If the Prob(some particular evidence|outcome) is 0, then the whole prob. becomes 0. If you see contradicting evidence, we can rule out that outcome.
											    Since we divide everything by P(Evidence), we can even get away without calculating it.
											    The intuition behind multiplying by the prior is so that we give high probability to more common outcomes, and low probabilities to unlikely outcomes. These are also called base rates and they are a way to scale our predicted probabilities.
											    How to Apply NaiveBayes to Predict an Outcome?

											    Just run the formula above for each possible outcome. Since we are trying to classify, each outcome is called a class and it has a class label. Our job is to look at the evidence, to consider how likely it is to be this class or that class, and assign a label to each entity. Again, we take a very simple approach: The class that has the highest probability is declared the "winner" and that class label gets assigned to that combination of evidences.

											    Fruit Example

											    Let's try it out on an example to increase our understanding: The OP asked for a 'fruit' identification example.

											    Let's say that we have data on 1000 pieces of fruit. They happen to be Banana, Orange or some Other Fruit. We know 3 characteristics about each fruit:

											    Whether it is Long
											    Whether it is Sweet and
											    If its color is Yellow.
											    This is our 'training set.' We will use this to predict the type of any new fruit we encounter.

											    Type           Long | Not Long || Sweet | Not Sweet || Yellow |Not Yellow|Total
											                 ___________________________________________________________________
													 Banana      |  400  |    100   || 350   |    150    ||  450   |  50      |  500
													 Orange      |    0  |    300   || 150   |    150    ||  300   |   0      |  300
													 Other Fruit |  100  |    100   || 150   |     50    ||   50   | 150      |  200
													             ____________________________________________________________________
														     Total       |  500  |    500   || 650   |    350    ||  800   | 200      | 1000
														                  ___________________________________________________________________
																  We can pre-compute a lot of things about our fruit collection.

																  The so-called "Prior" probabilities. (If we didn't know any of the fruit attributes, this would be our guess.) These are our base rates.

																   P(Banana)      = 0.5 (500/1000)
																    P(Orange)      = 0.3
																     P(Other Fruit) = 0.2
																     Probability of "Evidence"

																     p(Long)   = 0.5
																     P(Sweet)  = 0.65
																     P(Yellow) = 0.8
																     Probability of "Likelihood"

																     P(Long|Banana) = 0.8
																     P(Long|Orange) = 0  [Oranges are never long in all the fruit we have seen.]
																      ....

																      P(Yellow|Other Fruit)     =  50/200 = 0.25
																      P(Not Yellow|Other Fruit) = 0.75
																      Given a Fruit, how to classify it?

																      Let's say that we are given the properties of an unknown fruit, and asked to classify it. We are told that the fruit is Long, Sweet and Yellow. Is it a Banana? Is it an Orange? Or Is it some Other Fruit?

																      We can simply run the numbers for each of the 3 outcomes, one by one. Then we choose the highest probability and 'classify' our unknown fruit as belonging to the class that had the highest probability based on our prior evidence (our 1000 fruit training set):

																      P(Banana|Long, Sweet and Yellow)
																            P(Long|Banana) * P(Sweet|Banana) * P(Yellow|Banana) * P(banana)
																	        = _______________________________________________________________
																		                      P(Long) * P(Sweet) * P(Yellow)

																				          = 0.8 * 0.7 * 0.9 * 0.5 / P(evidence)

																					      = 0.252 / P(evidence)


																					      P(Orange|Long, Sweet and Yellow) = 0


																					      P(Other Fruit|Long, Sweet and Yellow)
																					            P(Long|Other fruit) * P(Sweet|Other fruit) * P(Yellow|Other fruit) * P(Other Fruit)
																						        = ____________________________________________________________________________________
																							                                          P(evidence)

																												      = (100/200 * 150/200 * 50/200 * 200/1000) / P(evidence)

																												          = 0.01875 / P(evidence)
																													  By an overwhelming margin (0.252 >> 0.01875), we classify this Sweet/Long/Yellow fruit as likely to be a Banana.

																													  Why is Bayes Classifier so popular?

																													  Look at what it eventually comes down to. Just some counting and multiplication. We can pre-compute all these terms, and so classifying becomes easy, quick and efficient.

																													  Let z = 1 / P(evidence). Now we quickly compute the following three quantities.

																													  P(Banana|evidence) = z * Prob(Banana) * Prob(Evidence1|Banana) * Prob(Evidence2|Banana) ...
																													  P(Orange|Evidence) = z * Prob(Orange) * Prob(Evidence1|Orange) * Prob(Evidence2|Orange) ...
																													  P(Other|Evidence)  = z * Prob(Other)  * Prob(Evidence1|Other)  * Prob(Evidence2|Other)  ...
																													  Assign the class label of whichever is the highest number, and you are done.

																													  Despite the name, Naive Bayes turns out to be excellent in certain applications. Text classification is one area where it really shines.

																													  Hope that helps in understanding the concepts behind the Naive Bayes algorithm.
