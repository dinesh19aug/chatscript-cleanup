# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:58:05 2019

@author: dines
"""
import pandas as pd


import util
from util import getTimeStamp, getChatText, getActivityId, extractCustomerChat, extractAgentChat
from util import extract_intent_textTeaser, extract_intent_summary, extract_intent, addgroupId
from util import merge_rake_genism, extract_intent_textTeaser_sanitized, extract_intent_summary_sanitized
from util import merge_rake_genism_sanitized


chat_df = pd.read_fwf(filepath_or_buffer="data\eGain Transcript sample.txt", delimiter="\n")

# Fix column Name
chat_df = chat_df.rename(index=str, columns={'Raw_chat': 'raw_chat'})

# Cut time stamp and create a new column
chat_df['timestamp'] = chat_df.apply(getTimeStamp, axis=1)
# Cut the chat text and create a new column
# chat_df['chat'] = chat_df.iloc[:, 0].str.slice(20).astype(str)
chat_df['chat'] = chat_df.apply(getChatText, axis=1)

# Drop the first column
chat_df = chat_df.drop(columns="raw_chat")

# Create a column activity_id and fill in column
chat_df['activity'] = chat_df.apply(getActivityId, axis=1)

# Remove Rows with Activity Id
chat_df = chat_df.dropna()

chat_df = chat_df[~chat_df['chat'].str.contains('Welcome to Wells Fargo.')]
chat_df = chat_df[~chat_df['chat'].str.contains('has ended the chat')]
chat_df = chat_df[~chat_df['chat'].str.contains('Chat has been initiated by customer')]
chat_df = chat_df[~chat_df['chat'].str.contains('You are now chatting with')]
chat_df = chat_df[~chat_df['chat'].str.contains('Hi David')]

#Cut Agent name and create a column agent_name
chat_df['cust_text']= chat_df.apply(extractCustomerChat, axis=1)

#Cut customer name and create column cust_name
chat_df['agent_text']= chat_df.apply(extractAgentChat, axis=1)

chat_df =chat_df.drop(columns="chat")

util.activity_id =0
chat_df['grouping'] = chat_df.apply(addgroupId, axis=1)

#drop timestamp
chat_df = chat_df.drop(columns="timestamp")

cust_only_chat=chat_df.groupby(['activity','grouping'])['cust_text'].apply(lambda x: '.'.join(x.astype(str))).reset_index()
agent_only_chat = chat_df.groupby(['activity','grouping'])['agent_text'].apply(lambda x: '.'.join(x.astype(str))).reset_index()

#Dropping None rows
import numpy as np
cust_only_chat['cust_text'] = cust_only_chat['cust_text'].replace(to_replace='None', value=np.nan).dropna(axis=0)
agent_only_chat['agent_text'] = agent_only_chat['agent_text'].replace(to_replace='None', value=np.nan).dropna(axis=0)
cust_only_chat['cust_text'] = cust_only_chat['cust_text'].replace(to_replace='None.None.None.', value=np.nan).dropna(axis=0)
cust_only_chat['cust_text'] = cust_only_chat['cust_text'].replace(to_replace='None.None.None.None', value=np.nan).dropna(axis=0)
agent_only_chat['agent_text'] = agent_only_chat['agent_text'].replace(to_replace='None.None.None.None', value=np.nan).dropna(axis=0)
agent_only_chat['agent_text'] = agent_only_chat['agent_text'].replace(to_replace='None.None.None', value=np.nan).dropna(axis=0)
cust_only_chat['cust_text'] = cust_only_chat['cust_text'].replace(to_replace='None.None', value=np.nan).dropna(axis=0)
agent_only_chat['agent_text'] = agent_only_chat['agent_text'].replace(to_replace='None.None', value=np.nan).dropna(axis=0)
cust_only_chat = cust_only_chat.dropna(axis=0)
agent_only_chat = agent_only_chat.dropna(axis=0)

#Dropping grouping column
cust_only_chat=cust_only_chat.drop(columns='grouping')
agent_only_chat=agent_only_chat.drop(columns='grouping')

#Merging all customer lines of an activity into 1 line
cust_only_chat = cust_only_chat.groupby(['activity'])['cust_text'].apply(lambda x: '.'.join(x.astype(str))).reset_index()
#Merging all agent lines of an activity into 1 line
agent_only_chat = agent_only_chat.groupby(['activity'])['agent_text'].apply(lambda x: '.'.join(x.astype(str))).reset_index()

#Merging cust and agent dataframe on activity Id and now we will have 1:1 mapping.
#Note: Agent only chats will sometime have more lines because customer initiated the chat and left without using
#any question
df_all = pd.merge(agent_only_chat,cust_only_chat.drop_duplicates(),
                   how='outer')

#Dropping all the rows where customer started the chat session but did not ask anything
#Now we have exact 1:1 mapping
df_all = df_all.dropna(axis=0)

#remove_words=['None.', 'Thank', 'you', 'You', 'thank','None', 'great', 'ok,', 'Ok,', 'OK,','ok.', 'Ok.', 'OK.','ok', 'Ok', 'OK','Thank You']
stop_words=pd.read_fwf(filepath_or_buffer="data\stop_words.txt", delimiter="\n")
remove_words=stop_words['list'].values.tolist()
pat = r'\b(?:{})\b'.format('|'.join(remove_words))
df_all['agent_text_sanitized'] = df_all['agent_text'].str.replace(pat, '')
df_all['cust_text_sanitized'] = df_all['cust_text'].str.replace(pat, '')

#Model 1 Using Rake: Add the intent to df_all
df_all['intent_rake'] = df_all.apply(extract_intent, axis=1)

#Model 2 Using Genism: Summarise the customer text
df_all['model_gensim_summary'] = df_all.apply(extract_intent_textTeaser, axis=1)
df_all['model_gensim_summary_sanitized'] = df_all.apply(extract_intent_textTeaser_sanitized, axis=1)
#Use rake to generate keywords using summarized text from model_genism_summary column
df_all['intent_genism'] = df_all.apply(extract_intent_summary, axis=1)
df_all['intent_genism_sanitized'] = df_all.apply(extract_intent_summary_sanitized, axis=1)

#For all single line cust text, summary is empty. Copy and replace empty intent_genism with intent_rake keyword
df_all['final_intent_keys']=df_all.apply(merge_rake_genism, axis=1)
df_all['final_intent_sanitized_keys']=df_all.apply(merge_rake_genism_sanitized, axis=1)
df_all.to_csv(r'C:\Dinesh\jup-notebook\intent.csv', index=None, header=True)




