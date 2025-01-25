# Customer-Loan-Propensity-Model-Python
A Customer Loan Propensity Model with data, including RFM analysis, customer segmentation, and actionable insights using Python Pandas NumPy Seaborn Matplotlib Sklearn

![alt text](https://github.com/gaptab/Customer-Loan-Propensity-Model-Python/blob/main/visualization.png)

A propensity model is a predictive model used to estimate the likelihood (or propensity) of a specific event or behavior occurring. It typically assigns a probability score to each individual or entity, indicating their likelihood of taking a desired action.

RFM stands for Recency, Frequency, and Monetary. It is a popular method used to analyze and segment customers based on their purchasing behaviors. RFM analysis helps businesses understand customer behavior, predict future purchases, and identify high-value customers for targeted marketing.

Here's what each term means:

**Recency (R**):

Refers to how recently a customer has made a purchase or engaged with the business.
In the context of a loan propensity model, it can be thought of as how recently the customer has interacted with the loan offerings or applied for a loan.
A higher recency score means the customer has recently interacted with the business, indicating they may be more likely to make another purchase or take out a loan.

**Frequency (F)**:

Refers to how often a customer makes a purchase or engages with the business.
In a loan model, this could indicate how often a customer has taken loans or interacted with financial products in the past.
A higher frequency score suggests a loyal customer who engages with the business regularly.

**Monetary (M)**:

Refers to how much money the customer spends or contributes to the business.
In this case, it can indicate how much a customer has borrowed or how frequently they repay their loans.
Customers with high monetary values are generally considered more valuable because they contribute more revenue.
RFM Segmentation
Once you have these three metrics (Recency, Frequency, and Monetary), you can segment your customers based on their scores. Here's how:

Recency: Customers who bought recently are likely to buy again. Customers who haven't bought in a long time might be considered "at risk."
Frequency: Customers who purchase frequently are loyal and likely to engage again, while infrequent buyers might need targeted attention.
Monetary: High-value customers who spend a lot should be prioritized for future offers or upselling opportunities.

RFM segments customers into categories based on their behavior:

**Low RFM score**: These are customers who are less engaged and less valuable.
**Medium RFM score**: These are regular customers who occasionally engage with the business.
**High RFM score**: These are highly engaged, loyal customers who are valuable and might be the best candidates for new offers.

**Decile** 
A Decile is a statistical term that divides the dataset into 10 equal parts, with each part representing 10% of the data.

**Top Decile (1st Decile)**: Refers to the top 10% of the customers, based on a certain metric (in this case, loan propensity). These customers are considered the most likely to engage or take a loan.
**Bottom Decile (10th Decile)**: Refers to the bottom 10% of the customers, who have the lowest probability of taking a loan.
In the model, deciles are often used to rank customers based on their predicted likelihood of taking a loan. 

For example:
Customers in the 1st decile have the highest predicted probability of taking a loan.
Customers in the 10th decile have the lowest predicted probability.
Why Deciles?
Using deciles helps businesses prioritize high-value customers (top deciles) and optimize marketing efforts by targeting specific customer groups based on predicted behaviors.

444


