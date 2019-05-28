from gensim.summarization import summarize
from rake_nltk import Rake, Metric


def getTimeStamp(row):
    first_20col = row['raw_chat']
    if (first_20col.count("Activity") == 0):
        return first_20col[0:19]


def getChatText(row):
    raw_chat = row['raw_chat']
    if (raw_chat.count("Activity") == 0):
        return raw_chat[20:]
    else:
        return raw_chat


def getActivityId(row):
    global activity_id
    chat = row['chat']
    if (chat.count("Activity") == 0):
        return activity_id
    else:
        activity_id = chat[13:]


def extractCustomerChat(row):
    chat = row['chat']
    if (chat.count("David L.:") == 0):
        return chat[(chat.index(':') + 1):]
    else:
        return


def extractAgentChat(row):
    chat = row['chat']
    if (chat.count("David L.:") != 0):
        return chat[(chat.index(':') + 1):]
    else:
        return


def extract_intent_textTeaser(row):
    #text = " Questions about refinancing my auto loan, Morning David :). I recently went to the dealer to actually trade my car in (I currently have a very high payment)  - they said I was approved through  guys at a much lower rate (cleaned up my credit) . but I am considering just refinancing this one. because it's only 2 years old , Do  guys have to do a hard inquiry on my credit since  already have my info? . I really don't want another 6 point hit.  I am cleaning up my credit, but 12 pts in 1 month isn't going to be a good thing.  kinda taking 2 steps forward, 3 steps back situation, So if  were in my position.. what would  suggest?, when they do a refinance, they don't add months correct?  they refinance for the remaining months? . I might have to buy a second car. so I am trying to lower my payment at the very least, I can't do that with  though. I have to go through the website, right?"
    #text=" my application was approved. what do I do next?. if i was approved for a certain amount would I be able to go to a dealership?. and what would I take with me to show i have been pre approved for a certain amount. how can i best move forward with getting the check ahead of time and securing the vehicle before visiting the dealership?.  for your help"
    custText = row['cust_text']
    #print(row['activity'])
    if (row['activity'] != '1420820' and row['activity'] != '1554108' and row['activity'] != '1662813' and row[
        'activity'] != '80445'):
        try:
            t=(summarize(custText,ratio=0.7,word_count=None,split=False))
            return t
        except ValueError as e:
            #print( " Issue with: " + row['activity'])
            return 'None'


def extract_intent_textTeaser_sanitized(row):
    #text = " Questions about refinancing my auto loan, Morning David :). I recently went to the dealer to actually trade my car in (I currently have a very high payment)  - they said I was approved through  guys at a much lower rate (cleaned up my credit) . but I am considering just refinancing this one. because it's only 2 years old , Do  guys have to do a hard inquiry on my credit since  already have my info? . I really don't want another 6 point hit.  I am cleaning up my credit, but 12 pts in 1 month isn't going to be a good thing.  kinda taking 2 steps forward, 3 steps back situation, So if  were in my position.. what would  suggest?, when they do a refinance, they don't add months correct?  they refinance for the remaining months? . I might have to buy a second car. so I am trying to lower my payment at the very least, I can't do that with  though. I have to go through the website, right?"
    #text=" my application was approved. what do I do next?. if i was approved for a certain amount would I be able to go to a dealership?. and what would I take with me to show i have been pre approved for a certain amount. how can i best move forward with getting the check ahead of time and securing the vehicle before visiting the dealership?.  for your help"
    custText = row['cust_text_sanitized']
    #print(row['activity'])
    if (row['activity'] != '1420820' and row['activity'] != '1554108' and row['activity'] != '1662813' and row[
        'activity'] != '80445'):
        try:
            t=(summarize(custText,ratio=0.7,word_count=None,split=False))
            return t
        except ValueError as e:
            #print( " Issue with: " + row['activity'])
            return 'None'

def extract_intent_summary(row):
    custText = row['model_gensim_summary']
    #print(row['activity'])
    if (row['activity'] != '1420820' and row['activity'] != '1554108' and row['activity'] != '1662813' and row[
        'activity'] != '80445'):
        # r = Rake(min_length=2, max_length=7)
        #r = Rake(min_length=2, max_length=7, ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
        r = Rake(min_length=2, max_length=7,ranking_metric=Metric.WORD_DEGREE)
        r.extract_keywords_from_text(custText)
        result = r.get_ranked_phrases()
        #print (r.get_ranked_phrases_with_scores())

        return result[:4]
    else:
        return

def extract_intent_summary_sanitized(row):
    custText = row['model_gensim_summary_sanitized']
    #print(row['activity'])
    if (row['activity'] != '1420820' and row['activity'] != '1554108' and row['activity'] != '1662813' and row[
        'activity'] != '80445'):
        # r = Rake(min_length=2, max_length=7)
        #r = Rake(min_length=2, max_length=7, ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
        r = Rake(min_length=2, max_length=7,ranking_metric=Metric.WORD_DEGREE)
        r.extract_keywords_from_text(custText)
        result = r.get_ranked_phrases()
        #print (r.get_ranked_phrases_with_scores())

        return result[:4]
    else:
        return


activity_id = 0
group_id=1
flag_none=False


def extract_intent(row):
    custText=row['cust_text']
    # print(row['activity'])
    if(row['activity'] != '1420820' and row['activity'] !='1554108' and row['activity'] !='1662813' and row['activity']!='80445' ):
        #r = Rake(min_length=2, max_length=7)
        #r = Rake(min_length=1, max_length=7,ranking_metric=Metric.DEGREE_TO_FREQUENCY_RATIO)
        #r = Rake(min_length=2, max_length=7,ranking_metric=Metric.WORD_DEGREE)
        r = Rake(min_length=2, max_length=7, ranking_metric=Metric.WORD_FREQUENCY)
        r.extract_keywords_from_text(custText)
        result=r.get_ranked_phrases()
        #print (r.get_ranked_phrases_with_scores())

        return result[:4]
    else:
        return


def addgroupId(row):
    global activity_id
    global group_id
    global flag_none
    if(activity_id == 0):
        activity_id=row['activity']

    if(activity_id ==row['activity']):
        if(row['cust_text'] == None ):
            if flag_none==False:
                group_id = group_id + 1
                flag_none = True
                return group_id
            else:
                return group_id

        else:
            if flag_none == True:
                group_id=group_id+1
                flag_none =False
                return group_id
    else:
        group_id = 1
        activity_id = 0
    return group_id

def merge_rake_genism(row):
    intent_genism = row['intent_genism']
    print(row['activity'])
    if (row['activity'] != '1420820' and row['activity'] != '1554108' and row['activity'] != '1662813' and row[
        'activity'] != '80445'):
        if len(intent_genism) == 0:
            return row['intent_rake']
        else:
            return intent_genism

def merge_rake_genism_sanitized(row):
    intent_genism = row['intent_genism_sanitized']
    print(row['activity'])
    if (row['activity'] != '1420820' and row['activity'] != '1554108' and row['activity'] != '1662813' and row[
        'activity'] != '80445'):
        if len(intent_genism) == 0:
            return row['intent_rake']
        else:
            return intent_genism