import pandas as pd
from functools import reduce
from bs4 import BeautifulSoup
from tqdm import tqdm
import helpscout

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

'''
Download emails from a helpscout account.
Output will be saved as emails.{mailbox_id}.csv

Input: api_key - helpscout API key
'''
def download_helpscout_emails(api_key):
    hs = helpscout.Client()
    hs.api_key = api_key
    
    print("Downloading emails. This might take a while...")
    for mailbox in hs.mailboxes():
        print("Processing mailbox" + mailbox.id)
        filename = "emails." + str(mailbox.id) + ".csv"
        rows = []
        
        with open(filename, 'w') as outfile:
            outfile.write("mailbox_id@@,@@convo_id@@,@@type@@,@@customer_email@@,@@convo_created_by_email@@,@@created_at@@,@@closed_at@@,@@status@@,@@subject@@,@@thread_created_at@@,@@thread_created_by@@,@@body\n")

        for conversation in hs.conversations_for_mailbox(mailbox.id):
            conversation = hs.conversation(conversation.id)

            for thread in conversation.threads:
                if thread.type in ('message', 'customer') and thread.body is not None:
                    row = [mailbox.id, conversation.id, thread.type, conversation.customer["email"], \
                           conversation.createdby["email"], conversation.createdat, conversation.closedat, conversation.status, \
                           conversation.subject, thread.createdat, thread.createdby["email"], thread.body.replace('\n', ' ')]

                    row = '@@,@@'.join(str(x) for x in row)
                    rows.append(row)

            if len(rows) > 1000:
                with open(filename, 'a') as outfile:
                    outfile.write("\n".join(rows))
                rows = []
        
        if len(rows) > 0:
            with open(filename, 'a') as outfile:
                outfile.write("\n".join(rows))
            rows = []

'''
Prepare the emails from "file":
 - strip HTML and extract text
 - convert dates to timestamps
 - replace null values with 0 (for "closed_at" field)
 - drop rows where the message body is empty
 - calculate sentiment scores
 
Result is saved in the outfile
Params: 
- file(str) - where the data is
- outfile(str) - where the output is saved
- contacts - DataFrame (account_id, email)
- websites - DataFrame (account_id, website)
'''
def transform_data(file, destination, contacts = None, websites = None):
    print("Loading data...")
    
    # Read data
    df = pd.read_csv(file, sep="@@,@@")
    print("Data succesfully loadeed. Memory usage: %s" % "{:,}".format(df.memory_usage(index=True, deep=True).sum()))
    
    # Replace null values with 0 on empty dates and convert them to timestamps
    df["created_at"] = pd.to_datetime(df['created_at']).values.astype("int64") / 1000000000
    df["thread_created_at"] = pd.to_datetime(df['thread_created_at']).values.astype("int64") / 1000000000
    
    df["closed_at"].fillna(0, inplace=True)
    df["closed_at"] = pd.to_datetime(df['closed_at']).values.astype("int64") / 1000000000
    
    # Strip HTML tags and extract text from message bodies
    print("Extracting text from HTML")
    for i in tqdm(range(df.shape[0])):
        soup = BeautifulSoup(df["body"][i], 'html.parser')
        df.at[i, "body"] = soup.get_text(separator=" ", strip=True)
    
    # Calculate sentiment scores
    df["positive_sentiment"] = 0.
    df["negative_sentiment"] = 0.
    df["compound_sentiment"] = 0.
    
    # Drop values where the body is empty (after striping HTML tags)
    df.dropna(subset=['body'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("Analyzing sentiment polarities")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    for i in tqdm(range(df.shape[0])):
        scores = sentiment_analyzer.polarity_scores(df["body"][i])
        df.at[i, "positive_sentiment"] = scores["pos"]
        df.at[i, "negative_sentiment"] = scores["neg"]
        df.at[i, "compound_sentiment"] = scores["compound"]
    
    # Match emails to accounts
    df["account_id"] = None
    df["invalid_email"] = False

    for index, row in df.iterrows():
        if not row["customer_email"] == row["customer_email"]:
            df.at[index, "invalid_email"] = True
            continue

        if contacts is not None:
            value = contacts.loc[contacts["email"] == row["customer_email"]]["account_id"]
        elif websites is not None:
            value = websites.loc[websites["website"] == row["customer_email"].split('@')[1]]["account_id"]
            
        if not (value.empty):
            df.at[index, "account_id"] = value.iloc[0]
    
    # Save result
    print("Saving output")
    df.to_csv(destination, index=False)
    
'''
Generated features:
- Overall sentiment score
- Sentiment from last email
- Sentiment from last 3 emails
- Sentiment from last 5 emails
- Rate of unresponsiveness

Inputs:
- df(DataFrame) - the data
- outfile(str) - where to save the data
- key(str) - what to group the data by (options are "account_id" and "customer_email")
'''
def generate_features(df, outfile, key = "customer_email"):
    # drop values where the key is null
    df.dropna(subset=[key], inplace=True)
    
    # get unresponsive rate
    last_email = df.groupby(key, group_keys=False).apply(lambda x: x.nlargest(1, "thread_created_at")).loc[:, [key]]
    last_customer_email = df[df["type"] == "customer"].groupby(key, group_keys=False) \
        .apply(lambda x: x.nlargest(1, "thread_created_at")).loc[:, [key]]
    
    unresponsive_rate = last_email.copy()
    unresponsive_rate["unresponsive_rate"] = 0
    
    for index, row in unresponsive_rate.iterrows():
        last_customer_email_index = last_customer_email.index[last_customer_email[key] == row[key]]
        if last_customer_email_index.empty:
            continue

        unresponsive_rate.at[index, "unresponsive_rate"] = index - last_customer_email_index[0]
    
    # drop emails that couldn't be matched
    df.dropna(subset=[key], inplace=True)
    
    # Keep only customer sent emails (dropping emails sent by us)
    df = df[df["type"] == "customer"]
    df.reset_index(drop=True, inplace=True)
    
    # Sort, extract and calculate averages
    mean = df.groupby([key], as_index=False)["compound_sentiment"].mean()
    last_1t = df.groupby(key, group_keys=False).apply(lambda x: x.nlargest(1, "thread_created_at")) \
        .groupby([key], as_index=False)["compound_sentiment"].mean()
    last_3t = df.groupby(key, group_keys=False).apply(lambda x: x.nlargest(3, "thread_created_at")) \
        .groupby([key], as_index=False)["compound_sentiment"].mean()
    last_5t = df.groupby(key, group_keys=False).apply(lambda x: x.nlargest(5, "thread_created_at")) \
        .groupby([key], as_index=False)["compound_sentiment"].mean()
    last_1c = df.groupby(key, group_keys=False).apply(lambda x: x.nlargest(1, "created_at")) \
        .groupby([key], as_index=False)["compound_sentiment"].mean()
        
    values = reduce(lambda left, right: pd.merge(left, right, on=key), [mean, last_1t, last_3t, last_5t, last_1c, unresponsive_rate])
    values.columns = [key, "sentiment_average", "sentiment_last_thread", "sentiment_last_3_threads", "sentiment_last_5_threads", "sentiment_last_convo", "unresponsive_rate"]
    
    # Save result
    print("Saving output")
    values.to_csv(outfile, index=False)