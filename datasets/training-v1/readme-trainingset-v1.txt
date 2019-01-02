========================

OffensEval 2019: Identifying and Categorizing Offensive Language in Social Media (SemEval 2019 - Task 6)
Training data 
v 1.0: November 28 2018
https://competitions.codalab.org/competitions/20011

========================

1) DESCRIPTION

The file offenseval-training-v1.tsv contains 13,240 annotated tweets. 

The dataset was annotated using crowdsourcing. The gold labels were assigned taking the agreement of three annotators into consideration. No correction has been carried out on the crowdsourcing annotations. 

The file offenseval-annotation.txt contains a short summary of the annotation guidelines.

Twitter user mentions were substituted by @USER and URLs have been substitute by URL.

Each instance contains up to 3 labels each corresponding to one of the following sub-tasks:

- Sub-task A: Offensive language identification; 

- Sub-task B: Automatic categorization of offense types;

- Sub-task C: Offense target identification.	

2) FORMAT

Instances are included in TSV format as follows:

ID	INSTANCE	SUBA	SUBB	SUBC 

Whenever a label is not given, a value NULL is inserted (e.g. INSTANCE	NOT	NULL	NULL)

The column names in the file are the following:

id	tweet	subtask_a	subtask_b	subtask_c

The labels used in the annotation are listed below.

3) TASKS AND LABELS

(A) Sub-task A: Offensive language identification

- (NOT) Not Offensive - This post does not contain offense or profanity.
- (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offense

In our annotation, we label a post as offensive (OFF) if it contains any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct. 

(B) Sub-task B: Automatic categorization of offense types

- (TIN) Targeted Insult and Threats - A post containing an insult or threat to an individual, a group, or others (see categories in sub-task C).
- (UNT) Untargeted - A post containing non-targeted profanity and swearing.

Please note that now targeted threats (TTH) have been merged with targeted insults (TIN) and are listed under Targeted Insult and Threats (TIN). The TTH label present in the trial set is not included in this training set and will not be included in the test set.

Posts containing general profanity are not targeted, but they contain non-acceptable language.

(C) Sub-task C: Offense target identification

- (IND) Individual - The target of the offensive post is an individual: a famous person, a named individual or an unnamed person interacting in the conversation.
- (GRP) Group - The target of the offensive post is a group of people considered as a unity due to the same ethnicity, gender or sexual orientation, political affiliation, religious belief, or something else.
- (OTH) Other – The target of the offensive post does not belong to any of the previous two categories (e.g., an organization, a situation, an event, or an issue)

Please note that now organization are listed under Other (OTH). The ORG label present in the trial set is not included in this training set and will not be included in the test set.

Label Combinations

Here are the possible label combinations in the OffensEval annotation.

-	NOT NULL NULL
-	OFF UNT NULL
-	OFF TIN (IND|GRP|OTH)

4) TRAINING PREDICTIONS (IMPORTANT!)

The OFFICIAL CodaLab competition (https://competitions.codalab.org/competitions/20011) will be open only in January. You will use it to upload your test set predictions which will be included in the official OffensEval ranks and shared task report. 

During the training stage, we created a PRACTICE CodaLab competition in which you can use to upload your predictions for each sub-task and evaluate your system. Here is the URL: https://competitions.codalab.org/competitions/20559?secret_key=5d5e72f8-bb17-49a6-9cf8-5827dafa2257

5) CREDITS

Task Organizers

Marcos Zampieri (University of Wolverhampton, UK)
Shervin Malmasi (Amazon, USA)
Preslav Nakov (Qatar Computing Research Insitute, Qatar)
Sara Rosenthal (IBM Research, USA)
Noura Farra (Columbia University, USA)
Ritesh Kumar (Bhim Rao Ambedkar University, India)

Contact

semeval-2019-task-6@googlegroups.com