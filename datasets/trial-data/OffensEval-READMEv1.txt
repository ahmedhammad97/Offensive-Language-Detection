========================

OffensEval 2019: Identifying and Categorizing Offensive Language in Social Media (SemEval 2019 - Task 6)
Trial data 
v 1.0: September 5 2018
https://competitions.codalab.org/competitions/20011

========================

1) DESCRIPTION

The file offenseval-trial.txt contains 320 annotated tweets. 

Each instance contains up to 3 labels each corresponding to one of the following sub-tasks:

- Sub-task A: Offensive language identification; 

- Sub-task B: Automatic categorization of offense types;

- Sub-task C: Offense target identification.	

2) FORMAT

Instances are included in TSV format as follows:

INSTANCE	SUBA	SUBB	SUBC 

Whenever a label is not given, a value NULL is inserted (e.g. INSTANCE	NOT	NULL	NULL)

The labels used in the annotation are listed below.

3) TASKS AND LABELS

(A) Sub-task A: Offensive language identification

- (NOT) Not Offensive - This post does not contain offense or profanity.
- (OFF) Offensive - This post contains offensive language or a targeted (veiled or direct) offense

In our annotation, we label a post as offensive (OFF) if it contains any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct. 

(B) Sub-task B: Automatic categorization of offense types

- (TIN) Targeted Insult - A post containing an insult to an individual, a group, or an organization (see categories in sub-task C).
- (TTH) Targeted Threat - A post containing a threat to an individual, a group, or an organization (see categories in sub-task C).
- (UNT) Untargeted - A post containing non-targeted profanity and swearing.

Posts containing general profanity are not targeted, but they contain non-acceptable language.

(C) Sub-task C: Offense target identification

- (IND) Individual - The target of the offensive post is an individual: a famous person, a named individual or an unnamed person interacting in the conversation.
- (GRP) Group - The target of the offensive post is a group of people considered as a unity due to the same ethnicity, gender or sexual orientation, political affiliation, religious belief, or something else.
- (ORG) Organization or Entity - The target of the offensive post is an organization (e.g., a company or an association) or an entity (e.g., a city, a country, a region, a continent, a location).
- (OTH) Other – The target of the offensive post does not belong to any of the previous three categories (e.g., a situation, an event, or an issue)

Label Combinations

Here are the possible label combinations in the OffensEval annotation.

-	NOT NULL NULL
-	OFF UNT NULL
-	OFF TIN (IND|GRP|ORG|OTH)
-	OFF TTH (IND|GRP|ORG|OTH)

IMPORTANT: The distribution of labels is the training and in the test data is likely to be different from the trial data (more offensive content).

4) CREDITS

Task Organizers

Marcos Zampieri (University of Wolverhampton, UK)
Shervin Malmasi (Harvard Medical School, USA)
Preslav Nakov (Qatar Computing Research Insitute, Qatar)
Sara Rosenthal (IBM Research, USA)
Noura Farra (Columbia University, USA)
Ritesh Kumar (Bhim Rao Ambedkar University, India)

Contact

semeval-2019-task-6@googlegroups.com