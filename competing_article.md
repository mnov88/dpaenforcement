PUBLISHED COMPETING ARTICLE -- !!! READ IN FULL !!

- [View **PDF**](https://www.sciencedirect.com/science/article/pii/S2212473X25000598/pdfft?md5=ab3e92bf7bd910f58b6233f282a2d22e&pid=1-s2.0-S2212473X25000598-main.pdf)

[![Elsevier](https://www.sciencedirect.com/eu-west-1/prod/39d89e3c832485a52ada0e211f012a3cc7ab704b/image/elsevier-non-solus.svg)](https://www.sciencedirect.com/journal/computer-law-and-security-review "Go to Computer Law & Security Review on ScienceDirect")

## Computer Law & Security Review

[Volume 59](https://www.sciencedirect.com/journal/computer-law-and-security-review/vol/59/suppl/C "Go to table of contents for this volume/issue"), November 2025, 106187

## A semantic approach to understanding GDPR fines: From text to compliance insights

[https://doi.org/10.1016/j.clsr.2025.106187](https://doi.org/10.1016/j.clsr.2025.106187 "Persistent link using digital object identifier") [Get rights and content](https://s100.copyright.com/AppDispatchServlet?publisherName=ELS&contentID=S2212473X25000598&orderBeanReset=true)

Under a Creative Commons [license](http://creativecommons.org/licenses/by-nc-nd/4.0/)

Open access

- [Next article in issue](https://www.sciencedirect.com/science/article/pii/S2212473X25000793)

## 1\. Introduction

The General Data Protection Regulation (GDPR) has reshaped Europe’s data protection landscape since May 2018. Its wide jurisdiction and robust enforcement mechanisms have generated substantial decisions and fines, providing opportunities to understand regulatory priorities and compliance dynamics across sectors and member states. Initially, authorities focused on monitoring market developments and addressing legacy cases that the GDPR had not yet governed. This phase was characterized by a relatively low number of fines and modest amounts. However, after this initial “orientation phase”, data protection authorities (DPAs) significantly intensified enforcement over time . Several record fines and landmark cases have garnered significant media attention. Notable examples include the €1.2 billion fine imposed on Meta Platforms Ireland Limited (2023) for an inadequate legal basis for data processing, as well as the €746 million fine against Amazon Europe Core S.r.l. (2021) due to non-compliance with general data processing principles. However, these high-profile fines are just the tip of the iceberg; hundreds of documented cases provide a substantial overview of GDPR enforcement, as well as numerous cases that are not disclosed by data protection authorities or made public in other ways. A thorough understanding of GDPR enforcement is crucial, as it provides valuable insights for stakeholders, policymakers, and researchers alike. Unpredictable enforcement patterns undermine legal certainty for both organizations and individuals.

Although significant efforts have been made to analyze these enforcement actions, most existing studies rely on manually curated datasets and domain-specific knowledge to extract meaningful patterns from legal decisions , , .

In this paper, we propose an Explainable Artificial Intelligence (XAI) framework that enables automated, scalable, transparent, and reproducible analysis of GDPR enforcement decisions, particularly important in light of due process requirements and the right to contest decisions in legal contexts. Our framework integrates natural language processing (NLP) with semantic and topic modeling techniques to uncover latent structures in over 1900 cases from the GDPR Enforcement Tracker, maintained by CMS.Law .

Our study is guided by the following research questions, with each addressed using a specific methodological approach.

- •
	**RQ1: How are fines for GDPR violations distributed across European countries?***Approach: Automated extraction of fine amounts, DPAs, and jurisdiction metadata to statistically quantify disparities. Cross-country comparisons contextualize variations in enforcement intensity, addressing debates about regulatory divergence and the efficacy of the one-stop-shop mechanism. (Sections**,**).*
- •
	**RQ2: What is the relationship between violation severity and monetary fines?***Approach: Semantic analysis (Keyness and WordNet) at the word level of fines by amount classes and tier classes. (Sections**,**,**)*
- •
	**RQ3: Can we identify structural patterns within the text data considering fine amounts and violation severity?***Approach: Network analysis (WordNet) and structural topic modeling (STM) to link cited GDPR articles, violation arguments, and fine tiers. (Sections**,**).*
- •
	**RQ4: What are the most common factors triggering GDPR enforcement?***Approach: Topic modeling (STM) and keyness analysis to identify prevalent themes (e.g., “video surveillance misuse”) and their correlation with tier/amount categories, highlighting systemic compliance gaps. (Sections**,**,**)*

By addressing these questions, we present a scalable and interpretable NLP-based framework that not only replicates key findings from earlier manually curated studies on GDPR enforcement, but also identifies more nuanced and previously unexamined patterns. Our core contribution is methodological: we bridge legal scholarship with automated legal analytics by providing computational tools that are both rigorous and transparent. Prioritizing explainability ensures that model outputs remain intelligible and trustworthy for legal experts, regulators, and policymakers, an essential requirement in governance and legal contexts.

## 2\. The GDPR framework

Adopted on April 14, 2016, the GDPR became fully enforceable on May 25, 2018 . It applies to any entity processing the personal data of individuals in the EU, regardless of the entity’s geographic location (Article 3). This extraterritorial scope has established the GDPR as one of the most influential and far-reaching data protection laws worldwide.

The Regulation comprises 11 chapters and contains 99 articles. The basic principles under Art. 5 are:

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr1.jpg)

Download: Download high-res image (269KB)

- •
	Lawfulness, fairness, and transparency: data processing must be conducted in a lawful and transparent manner, ensuring fairness to the data subject.
- •
	Purpose limitation: data should only be processed for specific, legitimate purposes that have been clearly communicated to the data subject.
- •
	Data minimization: only the data necessary for the stated purposes should be collected and processed.
- •
	Accuracy: personal data must be kept accurate and up-to-date.
- •
	Storage limitation: data should be stored only as long as necessary for its intended purpose.
- •
	Accountability: the data controller is responsible for demonstrating compliance with these GDPR principles.

Non-compliance with 44 of the GDPR 99 articles may result in significant fines.

Regarding the severity of the violation, Art. 83 outlines two distinct levels of fines, reflecting the provisions of the GDPR that the EU considers most critical. The first level addresses breaches of Articles 8, 11, 25–39, 41, 42, and 43, with a maximum penalty of 10,000,000 Euros or 2 percent of total global annual turnover, whichever is greater (, Tier *a*). The second level — associated with more serious violations — covers breaches of Articles 5, 6, 7, 9, 12–22, 44–49, and 58, and carries a maximum fine of 20,000,000 Euros or 4 percent of global annual turnover (see , Tier *b*).

In what follows, we refer to these two categories as *Tier a* (less severe violations) and *Tier b* (more severe violations).

Although European DPAs have considerable freedom to determine their fine model, authorities are guided by predefined legal criteria (Article 83). These criteria encompass various factors, including deliberate violations, a lack of efforts to mitigate harm, and failure to cooperate with authorities, which may increase the severity of the penalty.

Compliance responsibilities apply to “any natural or legal person, public authority, agency, or other body” involved in processing personal data (Article 4). This definition encompasses both individuals and corporate entities, ensuring the universal application of GDPR obligations. Consequently, all parties — from individuals to large corporations — must follow the same core GDPR principles, including data minimization, lawful processing, and transparency, and face comparable penalties for non-compliance.

## 3\. Related work

A foundational overview of the GDPR, including its structure and objectives, is provided by Hoofnagle et al. . Zaguir et al. provide a comprehensive literature review on the implementation and sustainability of GDPR compliance. They emphasize barriers and enablers to effective enforcement and identify implementation gaps while suggesting new research directions. Several studies have focused on the GDPR’s impact on organizations and global technological development, analyzing the dual nature of the regulation as both a challenge and an opportunity for innovation and risk management, including , .

A key concern is the level of awareness among small businesses about obligations under the GDPR. In response, the European Data Protection Board has issued a dedicated Data Protection Guide supporting SMEs through accessible and practical compliance guidance . This aims to improve understanding and practical implementation within smaller entities. Notably, GDPR obligations also extend to individuals and private associations, as outlined in Article 4, broadening scope and raising complex issues around violation assessment and sanction proportionality. These issues are further explored in subsequent sections. From an enforcement perspective, CMS summarizes European GDPR fines, showing issuing countries, fined companies, reasons (e.g., lack of legal basis), affected sectors, and trends including increasing penalties.

A foundational contribution by Wolff and Atallah analyzes 261 fines issued within the first two years of GDPR implementation. It reviews enforcement trends and key fines across jurisdictions to understand how regulatory approaches are changing

Although direct work on explainable AI (XAI) for GDPR fine motivations remains nascent, research in adjacent domains provides a robust technical basis. Research on privacy policy analysis has established effective pipelines for structuring and classifying unstructured legal text. For example, one AI-assisted approach combines fundamental NLP techniques — including tokenization, sentence splitting, Named Entity Recognition (NER), and lemmatization — with Support Vector Machine (SVM) classification for automatic content categorization (see ). Extending these classification approaches, recent research has progressed predictive analysis of GDPR fines. Expanding upon previous findings , a more detailed study of 294 enforcement cases applies text mining and regression analysis to forecast fine amounts . These automated predictions are critically evaluated within the broader context of algorithmic decision-making in the public sector. Pattern recognition and text categorization techniques, as applied to detecting potential data breach indicators , can directly identify key violations in fine texts.

A key step toward automated analysis is the extraction of structured information from regulatory texts, as demonstrated by research that uses NLP to generate structured representations of cross-jurisdictional regulations for similarity comparison . This highlights techniques essential for parsing fine motivations to identify core elements: specific violations, aggravating/mitigating factors, data types involved, and cited legal bases. Further evidence for feasibility of large-scale fine analysis comes from that offer a review of 856 fines issued since 2018 focusing on data flows while categorizing fines based on whether they resulted from organizational or technical issues. Using text summaries from the Enforcement Tracker, they conduct targeted word analysis specifically on fines related to technical issues, demonstrating practical application of text mining for categorization. Advanced IE techniques such as Semantic Role Labeling (SRL), combined with NER, applied to annotate GDPR transparency requirements as in , offer pathways for a fine-grained understanding of relationships within fine rationales, crucial for generating detailed explanations of these relationships.

Research on formalizing regulations and automating compliance checks establishes a direct conceptual bridge to XAI, such as methods to create machine-analyzable representations and techniques for automated compliance verification . Significant advancements include high-precision frameworks for identifying GDPR-relevant information in policies and checking its completeness against requirements (achieving 92.9% precision and 89.8% recall) .

## 4\. Data and methods

### 4.1. Dataset

As pointed out by Ruohonen and Hjerppe , to date, there is no centralized database maintained by the EU for GDPR enforcement decisions by Data Protection Authorities (DPAs).

To address this gap, various online data collections have recently emerged, initiated by non-governmental organizations, companies, and others , , . This study utilizes the GDPR Enforcement Tracker (ET) , maintained by CMS Law, an open-access database compiling publicly available information on GDPR fines and enforcement actions across the European Union member states.

The ET, though unofficial, is widely used in academic and policy research for analyzing data protection enforcement trends under the GDPR . Its structured and regularly updated dataset simplifies the processing of unstructured legal documents, facilitating the analysis of GDPR decision-making and enforcement patterns. While not exhaustive — missing some publicly disclosed fines — the ET is suitable for examining whether explainable AI can validate prior manual analyses, replicate GDPR enforcement findings, and enable deeper exploration.

The ET covers all 27 EU member states and 10 business sectors, plus the United Kingdom and Norway. For each fine, the following information is provided:

- •
	A unique Enforcement Tracker ID (ETid)
- •
	Country, decision date, fine amount, name of fined controller/ processor, and business sector
- •
	Cited GDPR articles
- •
	Generalized fine type
- •
	Source URL
- •
	Summary text

### 4.2. Data acquisition and preprocessing

We detail the preprocessing steps and feature engineering applied to the enforcement dataset. We employed the rvest R package to extract relevant information from the ET site table. This process involved web scraping the table’s content until June 30th, 2023, yielding a structured dataset for further analysis.

#### Data cleaning and standardization.

The *amount* column, containing textual descriptions of fines, was cleaned and standardized. We replaced various textual descriptions with corresponding numeric values, using the interval mean (). This standardization ensured consistency and facilitated analysis. ‘Unknown’ entries were replaced with NA and dropped.

#### GDPR invoked article extraction.

We developed a function to extract article numbers from the articles column. This function employed regular expressions to identify text following *Art.* and then removed unnecessary parts like *Art.* and *GDPR*. Digits were then extracted and converted to numeric values. When extraction errors occurred, NA values were assigned.

#### Enriching with additional data.

We enriched the enforcement tracker dataset with information on violation tiers. As established in Article 83 (Section ) and shown in , two fine levels exist: tier *a* and tier *b*. We merged those dataset (article, tier) with the main data frame using inner joins based on the *article* column. To assign tier classification to each fine, we applied the following rules: (1) if all articles cited in the fine were tier *a*, the fine was classified as tier *a*; (2) if all cited articles were tier *b*, the fine was classified as tier *b*; and (3) if articles from both tiers were cited, the fine was classified as tier *ab*.

#### Categorization of fines.

We created a factor variable *amount\_cl* to categorize fines based on their amount using a *case\_ when* statement with three categories: less than 10K€ ($<$ 10K€), from 10 to less than 100K€ (\[10–100)K€), and equal or more than 100K€ ($>$ 100K€). This categorization simplifies analysis and visualization of fine distributions.

#### Non-numeric amount conversion.

The presents original textual values from the amount column requiring manual conversion, along with their assigned numeric equivalents.

These processing steps transformed the raw data into a structured analytical format suitable for GDPR enforcement analysis.

Table 1. Conversion of textual descriptions to numeric values in amount column.

| Original textual value in amount | Assigned numeric Value |
| --- | --- |
| Unknown | NA |
| Fine amount between EUR 50 and EUR 800 | 425 |
| Fine amount between EUR 50 and EUR 100 | 75 |
| Fine amount between EUR 400 and EUR 600 | 500 |
| Fine amount between EUR 350 and EUR 1000 | 675 |
| Fine amount between EUR 300 and EUR 400 | 350 |
| Fine amount between EUR 100 and EUR 1.000 | 550 |
| Fine amount between EUR 200 and EUR 1.000 | 600 |
| Fine in six-digit amount | 500000 |
| Fine in five-digit amount | 50000 |
| Fine in four-digit amount | 5000 |
| Fine in three-digit amount | 500 |

### 4.3. Descriptive statistics

Preliminary analyses were first performed using frequency and relative frequency by country and sector, respectively. Due to the large differences in frequency values, a logarithmic scale was used on the $y$ -axis of the bar plot.

To explore differences between median fine and tier value, a violin plot was used. This type of boxplot displays the distribution of a continuous variable. Each violin plot consists of a density trace, showing the kernel density estimate of the data, and a boxplot superimposed on top. The dot within the box represents the median, while the horizontal lines represent the confidence interval (C.I.) for the median estimation. This type of plot is useful for visualizing skewed or multimodal distributions, as it provides a clear representation of the data’s shape and spread. We simply represented the evolution in time of fines’ number by tier using a cumulative count by date. All the graphs was made using R package ggplot2

### 4.4. Semantics analysis

#### 4.4.1. Corpus preprocessing

Corpus preprocessing was performed using the Semanticase web application . Semanticase integrates various open-source tools, such as R and Apache Tika.

Within Semanticase, several preprocessing steps were implemented to ensure data quality and consistency. These steps employed customized scripts built upon the quanteda R package .

##### Lowercasing.

All text was converted to lowercase to standardize word representation and facilitate subsequent analyses.

##### Cleaning transparency and bias control.

All analyses start from the verbatim text contained in the HTML; no additional CMS framing text is retained. Generic basic cleaning first removes punctuation, numbers, and 194 common English stop-words (conjunctions, prepositions, etc.) drawn from the Snowball list . Subsequently, an iterative human-in-the-loop refinement is carried out: after each provisional STM run, a legal-domain expert inspects the top topic terms and — using domain knowledge — decides whether further tokens (e.g., “GDPR”, “personal”) should be added to the stop-word list or merged via a synonym table. This cycle repeats until the expert is satisfied that no artefacts distort substantive legal concepts. Importantly, we apply neither minimum-frequency nor maximum-vocabulary thresholds, thereby preventing unintended loss of low-frequency but legally relevant expressions. The final custom stop-words list and synonyms table are provided as supplementary materials at [https://zenodo.org/records/16730822](https://zenodo.org/records/16730822).

#### 4.4.2. Word-level analysis

Following preprocessing, word-level analysis was performed, utilizing relative frequency analysis (keyness) and the WordNet lexical graph.

##### Keyness.

Keyness statistics are implemented in Semanticase via the textstat\_keyness function from the quanteda.textstats R package . These statistics measure the differential word occurrence between a target group and a reference group. Here, the target group comprised fines within a specific *tier* category, while the reference group encompassed combined word frequencies from all other fines. We applied Yates’ correction to the chi-squared statistic to mitigate potential biases arising from small sample sizes in individual documents. For a detailed discussion of keyness and its application in text analysis, refer to Bondi and Scott .

##### WordNet.

Using the Python NLTK library , we leveraged WordNet across three groups based on *tier* variable values (*a*, *ab*, *b*). This stratification allowed examination of how WordNet captures semantic relationships within each group, revealing patterns and associations not immediately apparent in the raw data.

#### 4.4.3. Topic model

This study utilized a customized Structural Topic Model (STM) implemented within the Semanticase web application for dataset analysis. The STM is a well-established technique for uncovering latent thematic structures within textual data. Crucially, the model does not rely on any pre-defined topic labels; instead, it automatically discovers and infers both the number of topics and their definitions from statistical patterns in the document collection starting from the co-occurrence word-document matrix. While researchers guide the process through preprocessing decisions (e.g., domain-specific stop word lists and synonym mappings), the topics themselves emerge without manual labeling.

To aid non-technical readers, we focus on a core concept: topic prevalence. The topic *prevalence* refers to the degree to which a particular topic is represented within a document or across the corpus. It measures the prominence of a topic in a text, and using STM, we assess per-topic whether there is a stronger or weaker association with a specific category.

STM defines the probability of document-topic-covariate association (category). This enables some documents in a topic to be more strongly (high probability) or weakly (low probability) associated with a specific category. Thus, documents sharing a covariate value may show higher/lower probability for specific topics.

The customized Semanticase implementation builds upon the stm R package as a foundation while incorporating key enhancements.

##### Enhanced input data.

Semanticase extends standard STM by incorporating single words, bigrams (two consecutive words), and trigrams (three consecutive words). This captures complex relationships between words and enables better contextual understanding.

##### Data-driven topic number estimation.

Unlike traditional STM implementations relying on subjective parameter choices, Semanticase employs a data-driven approach to estimate the optimal number of topics (K). This approach uses Singular Value Decomposition (SVD) to tailor topics to the dataset, reducing bias risk.

##### Improved visualization.

Semanticase generates interactive HTML visualizations representing identified topics. These enable comprehensive topic exploration, allowing researchers to investigate relationships between N-grams and themes.

The customized STM within Semanticase additionally enabled estimation of topic prevalence and content dependence based on covariates. We leveraged *amount\_cl* and *tier* covariates to examine whether and how thematic structures varied across classes. This analytical approach yielded insights into how *amount\_cl* and *tier* categories influence topic prevalence and content.

## 5\. Results

### 5.1. Descriptive statistics

Following the methodology in Section , we first present a high-level overview of GDPR fine distributions across countries and sectors from May 2018 through June 30, 2023. As confirmed by the CMS Enforcement Tracker Report , the ET dataset reveals several clear trends.

Regarding fine frequency, Spain leads significantly ($>$ 35%), followed by Italy ($∼$ 16%), Romania ($∼$ 8%), and Germany ($∼$ 5%). However, total monetary penalties show a different pattern: Ireland, Luxembourg, and France impose the highest cumulative fines despite accounting for relatively few cases (see , ).

GDPR fines scale with entity turnover. Consequently, low-amount fines do not necessarily correspond to less severe violations; more severe infringements, when committed by individuals or small organizations, often result in proportionally smaller financial penalties. This implies that the monetary value of a fine is not a reliable indicator of the severity of a violation, an important distinction explored in the following analysis. By examining the relationship between the severity of violation and the corresponding amount of fine, it becomes possible to assess the extent to which low-value fines are associated not only with minor infractions, but also with serious violations committed by smaller entities.

To this aim, we estimated the distribution of fine amounts by tier across all countries and sectors (). As defined, tier *a* corresponds to violations of lower severity, while tier *b* includes violations of higher severity. (). We created a summary “tier” variable for each fine based on cited GDPR articles (see Section . For fines citing multiple articles, we assigned: tier *a* if all violated articles corresponded to type *a*, tier *b* if all corresponded to type *b*, and tier *ab* if both article types were cited. The violin plots () show fine amount distributions (thousands of euros) for tiers *a*, *ab*, and *b*. The $x$ -axis displays fines on a logarithmic scale, ranging from approximately €0.1K to over €100,000K, while the $y$ -axis differentiates the three tiers. Tier *a* (in red) shows a right-skewed distribution with a long tail, indicating the presence of a few very large fines. Its median fine is around €10,000, with a relatively concentrated distribution. Tier *ab* (in green) shows a median slightly higher — between €10,000 and €20,000 — and is associated with a broader confidence interval. As its label implies, this tier represents a transitional group with mixed features from both a and b. Tier *a* (in blue) exhibits the highest concentration of lower-value fines (i.e., below €10,000), which are likely associated with cases involving small businesses, private associations, or individual actors. Nevertheless, the presence, albeit limited, of high-value fines within this tier is clearly observable. This is a noteworthy and somewhat unexpected finding, as it calls into question the assumption that the magnitude of a fine consistently correlates with the severity of the underlying violation. Furthermore, it suggests that a substantial proportion of serious infringements may be attributed to smaller entities.

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr2.jpg)

Download: Download high-res image (179KB)

, confirm these patterns. shows the temporal evolution of cumulative fines by tier across all sectors and countries, covering May 2018 to June 2023. As expected, fines increased across all three tiers. Tier *b* violations (most serious) represent the majority of fines, with growth rates from 2020 significantly exceeding tiers *a* and *ab*. Meanwhile, depicts the temporal evolution by amount class. The highest cumulative number occurs in the \[10–100)K€ class. Although the $<$ 10K€ class had the lowest cumulative total, it exhibited a high growth rate. This demonstrates that alongside the significant increase in serious violations (tier *b*), low-value fines ($<$ 10K€) also rose rapidly.

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr3.jpg)

Download: Download high-res image (148KB)

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr4.jpg)

Download: Download high-res image (178KB)

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr5.jpg)

Download: Download high-res image (268KB)

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr6.jpg)

Download: Download high-res image (314KB)

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr7.jpg)

Download: Download high-res image (343KB)

### 5.2. Semantics analysis

This section describes results from applying the semantic analysis methodology in Section . The goal is to provide insights into text meaning and structure, enabling deeper understanding of the main reasons for fines across tiers and amount classes.

We analyzed each group of the two categories, specifically: tier (a, ab, b) and class amount ($<$ 10K€, \[10–100)K€, and $>$ 100K€). Word-level analysis (Section ) gives us information about how significantly a word stands out within a particular target group, compared to the reference dataset (keyness). Moreover, WordNet lexical graphs enable capturing semantic relationships between words in each category group, revealing latent associations not immediately apparent. Integral to semantic analysis is topic modeling (Section ), a statistical method for identifying and extracting main themes (topics) from document collections.

#### 5.2.1. Word level analysis

By means of keyness analysis we identified about 150 key words for each class of the two categories. In , , the most frequent ones are reported for each class. contains the key words by amount of fines. Based on this analysis, we observe that the lowest fines ($<$ 10K€) involve violations related to video surveillance in public spaces. Fines primarily target private individuals. Another word connected to this class of fines is principle minimization; this principle, described in Art.5(c), is based on the idea that only necessary personal data should be collected and retained. Violating Art. 5 leads to more serious fines (tier b). For the \[10–100)K€ class, frequent words concern privacy violations of rape victims, specifically during sexual abuse cases. Other terms indicate failures to verify third-party identities or obtain consent for data sharing. For fines exceeding €100,000, key words relate to users and customers connected to digital platforms or telemarketing. Indeed, the highest fines target companies in the Media, Telecoms and Broadcasting sector (Meta Platforms, Google LLC, TikTok limited). Another keyness word is transparency (Arts. 13–14), requiring clear, accessible information about personal data processing.

Table 2. Keyness words by amount of fines (euros).

Table 3. Keyness words by tier.

| Tier a | Tier ab | Tier b |
| --- | --- | --- |
| Measures | organizational\_measures | Surveillance |
| Breach | technical\_organizational\_measures | Video |
| Technical | Measures | video\_surveillance |
| Security | protection\_impact | CCTV |
| technical\_organizational\_measures | protection\_impact\_assessment | principle\_minimization |
| adequate\_technical | Violations | installed\_video |
| unauthorized\_access | Users | violation\_principle\_minimization |
| adequate\_technical\_organizational | measures\_protect | installed\_video\_surveillance |
| measures\_protect | implement\_technical | legal\_basis |
| security\_measures | Individuals | public\_space |
| failed\_implement\_adequate\_technical | Risk | Violation |
| Protect | organizational\_measures\_protect | private\_individual |
| organizational\_measures\_ensure | ensure\_security | installed\_video\_surveillance\_ccvt |

#### 5.2.2. Topic model

What are the most frequent violations resulting in fines? Topic modeling provides answers. Based on the methodology in Section , we fit the topic model to 1719 documents and a 33,232-word dictionary, yielding 13 topics. shows expected topic proportions in the corpus. Topic 9 is the most prevalent (14.4%), followed by topic 8 (12.4%) and topic 11 (10.7%). We analyze Topic 9, which is characterized by the highest prevalence and is statistically more represented in the documents of the entire corpus than the other topics. First, examine the word cloud in —a visualization displaying words sized by their topic probability, enabling quick key term identification. Clearly, this topic concerns video surveillance, with terms relating to the data minimization principle appearing less prominently. As established in Section , this principle (Art. 5(c)) mandates collecting only necessary personal data, and violations incur tier b fines.

We verify the relationship between Topic 9 and fine severity (tier). shows Topic 9 prevalence in groups a, ab, and b with 95% confidence intervals. This topic is statistically more prevalent in tier b than tiers a or ab, with a 22.2% prevalence. This indicates Topic 9 more frequently occurs in tier b documents. This prevalence significantly exceeds corpus-wide prevalence (14.4%). Confidence intervals confirm this: tier b lower bound (17.5%) exceeds other groups’ upper bounds. For groups ab and a, the prevalence is 11.8% and 8.7% respectively, showing overlapping confidence intervals.

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr8.jpg)

Download: Download high-res image (90KB)

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr9.jpg)

Download: Download high-res image (66KB)

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr10.jpg)

Download: Download high-res image (111KB)

![](https://ars.els-cdn.com/content/image/1-s2.0-S2212473X25000598-gr11.jpg)

Download: Download high-res image (124KB)

Let us know look at the prevalence of topic 9 by class amount. Looking at it is clear that Topic 9 occurs more frequently in the $<$ 10K€ group, with a percentage of 22.3%, which is notably higher than the 14.4% found across the entire corpus. This distinctiveness in the $<$ 10K€ group is further supported by the confidence interval, where the minimum value (17.6%) exceeds the maximum intervals of the other two categories. Lastly, the expected percentages for the remaining groups are 10.7% for fines $>$ 100K€ and 9.4% for the \[10–100)K€ range. We finally analyzed the evolution of the expected prevalence of Topic 9 in the entire corpus over time (). It can be observed that the trend remains more or less stable, followed by an upward expected trend till August 2023.

The methodology used to analyze the dataset made it possible to identify the documents (text summaries of fines) in which the topics are represented in a statistically significant way. For topic 9, we identified 248 documents from which it is evident that fines have primarily been issued to private individuals, associations, commercial establishments, and small businesses due to improper use of CCTV. As an example, we comment the two most significant ones related to the fines with identification numbers ETid-635 and ETid-257. The first (ETid-635) of 3000€ was imposed by the Spanish DPA (AEPD) on a private individual who regularly rents apartments to tourists. He installed four video cameras on the floors and in the entrance area of the building incurring in a “violation of the principle of data minimization, as \[...\] monitoring was not necessary for the protection of the controller’s property. \[...\]. Furthermore, the controller \[...\] failed to inform the other residents of the building about the processing of their data. \[...\]. Also, the controller did not put up a sign in the building informing about the operation of the camera”. Thus he violated art. 5(1) (principle minimization) and art.13 (information to be provided where personal data are collected from the data subject). The second fine (ETid-257) of 3000€ was issued by the Spanish DPA (AEPD) on the restaurant operator LODEJU, S.L. for the violation of (Art. 5(1) c) and (Art. 13). The violation linked to the improper use of video surveillance underscores the risks associated with monitoring individuals and indicates that the standards for the appropriate use of invasive surveillance techniques may also apply to other technological innovations .

The second most prevalent topic in the entire corpus is Topic 8, with a percentage of 12.4%. (). The word cloud () highlights key GDPR principles, particularly the rights of the data subject, lawful data processing, and the need for a valid legal basis. Special attention is given to consent and compliance, which are critical to avoiding violations. These elements lie at the core of GDPR enforcement and are commonly referenced in investigations that result in fines. Based on , we observe considerable overlap among the confidence intervals; specifically, the upper bounds of the confidence intervals for groups ab (10.2%) and a (7.9%) exceed or are close to the expected value for group b, which is 9.4%. Thus, we do not observe a clear predominance of one group over another, as it was previously observed for topic 9. As for the class amount, shows that, dividing the entire corpus per class amount, topic 8 is more characteristic of fines pertaining to group \[10–100)K€, with a percentage of 19.2%. The expected proportion for fines exceeding 100K€ is 14.6%, while the expected percentage for the $<$ 10K€ group is 9.5%. Although this topic shows a distinct prevalence in the \[10–100)K€ group compared to the $<$ 10K€ one, we observe an overlap in the confidence intervals with the $>$ 100K€ class. Indeed, the upper bound of the confidence interval for this class is higher than the expected percentage of \[10–100)K€ group. To confirm these results, we refer to the two most representative documents (summary texts) among the 212 which are characteristic of this topic. They are related to the fines with identifiers ETid-32 and ETid-457. The first, belonging to class *b* and \[10–100)K€, concerns a 50,000€ fine imposed by the Berlin Supervisory Authority on a bank for violating Article 6 because it had processed ‘personal data of all former customers’ without permission. The second (ETid-457), belonging to *ab* and $>$ 100K€ classes is a 2,250,000€ fine issued by the French DPA (CNIL) to Carrefour France for non-compliance with general data processing principles in violating art. 5, 12, 13, 15, 17 (tier *b*) and art. 21, 32, 33 (tier *a*) ().

Regarding topic 11, which is the third most prevalent in the corpus with a percentage of 10.7%, (), the world cloud () underscores the obligation to adopt and sustain effective technical and organizational measures to protect personal data. It further highlights the potential risks of non-compliance, including the prospect of regulatory investigations. Topic 11 is prevalent in tier *a* with a median expected proportion of 31.8%, significantly higher than the median expected proportion of this topic across the entire corpus. This result is consistent with the allocation of fines based on severity (), which places violations related to organizational and technical measures within tier *a*. However, we also observe that class *ab* exhibits a high expected proportion (22.1%), and its confidence interval overlaps with that of the expected proportion for tier *a*. This evidence confirms the findings observed through the keyness analysis (). On the other hand, the tier *b* class shows clear irrelevance to the topic, with an expected median proportion of only 4.7%. Regarding the categorization by class amount, the confidence intervals are very wide and overlap, so there is no clear prevalence of one class over the others. Below, we describe the two fines related to the most representative documents of this topic among the 183 documents in which topic 11 is characteristic. A fine (identity number Etid-719) of 64,500€ was imposed by The Swedish DPA on Voice Integrate Nordic AB. This company violated Art. 32 because it failed to take appropriate technical and organizational measures to ensure an adequate level of security to protect personal data so that unauthorized persons could not access it. In this case the fine pertains to classes *a* and \[10–100)K €. Regarding the second document, it is about a fine of 800,000€ (ETid-827) issued on 22/07/2021 by the Italian Data Protection Authority (DPA) on Roma Capitale for GDPR violations related to parking meters introduced in 2018. The investigation revealed major shortcomings: lack of transparency about data processing, failure to designate the data processor, absence of proper agreements and instructions for service providers, undefined data retention periods, and inadequate security measures. The infringements belong to class *ab* because the company failed to comply with Art. 5, Art. 12, Art. 13 (tier *b*) and Art. 25, Art. 28, Art. 32 (tier *a*).

Finally, let us make some observations regarding the evolution over time of the expected proportion for topics 8 and 11. shows that the expected proportion for topic 8 has an increasing trend starting from May 2023, up to the forecast for August 2023. highlights instead, for the same period and for topic 11, a decreasing trend in the expected proportion.

## 6\. Conclusive remarks

This study introduces an Explainable Artificial Intelligence (XAI) framework that integrates natural language processing (NLP), semantic modeling, and structural topic modeling (STM) to analyze regulatory decision-making. Applied to a corpus of over 1900 enforcement decisions sourced from the CMS. Law GDPR Enforcement Tracker, the framework uncovers key patterns in enforcement practices and contributes to the methodological advancement of legal analytics. The findings demonstrate that explainable AI is capable of replicating key results from prior GDPR enforcement studies, thereby validating earlier manual approaches while offering more granular and interpretable analyses. Our research was guided by four research questions (RQs), outlined in Section along with the methodological approach used to address each of them. In response to RQ1, the analysis confirms pronounced disparities among EU member states, with Spain emerging as the most active enforcement authority. These findings prompt reflection on how variations in national legal traditions, institutional capacity, and public awareness might shape the application of GDPR across jurisdictions, highlighting a need for further comparative inquiry into enforcement asymmetries within the EU. Regarding the relationship between the monetary value of fines and the severity of the violation (RQ2), the proposed analysis offers some noteworthy insights. A substantial number of low-amount fines are linked to violations classified as severe under the Regulation’s tier system. Particularly, tier *b* breaches involving fundamental principles such as lawfulness, transparency, and the legal basis for data processing. While the fine amount is indeed calculated as a percentage of annual turnover, the key point is that these fines often correspond to serious breaches. As a result, many serious violations are committed by small entities such as small businesses, associations, or private individuals.

Over time, tier b fines have shown the most significant growth in frequency, accompanied by a sharp rise in low-value sanctions. This trend indicates a shift in enforcement focus, from exclusively high-profile cases to a broader range of granular and widespread privacy violations. The fact that many serious GDPR violations are committed by small entities raises important questions about the underlying causes of non-compliance. Rather than stemming from intentional misconduct, these violations may reflect a lack of resources, expertise, or awareness regarding data protection obligations. This highlights a potential imbalance in GDPR enforcement: while fines are scaled to the economic capacity of the offender, the severity of the violation may still point to systemic weaknesses in data governance among smaller actors. Consequently, enforcement strategies should not rely solely on punitive measures, but also incorporate targeted support, education, and simplified compliance tools to help smaller organizations meet their obligations. Such an approach would not only enhance fairness but also promote broader and more effective adherence to the GDPR across all sectors. Beyond these quantitative dimensions, the framework facilitates a nuanced understanding of enforcement content in addressing RQ3 and RQ4. The main aim here is to identify structural patterns linking fine amounts and violation severity. Moreover, the most common factors triggering GDPR enforcement are investigated. Network analysis (WordNet) and STM techniques uncover consistent thematic patterns across decisions, confirming the centrality of principles such as valid consent, the rights of data subjects, and the implementation of appropriate technical and organizational safeguards. The recurrence of these themes across both high- and low-value fines reinforces their foundational role in GDPR compliance and their prominence in regulatory scrutiny.

Particularly noteworthy is the frequency with which violations related to video surveillance appear in the dataset. These cases often involve improperly configured systems that, while intended for security, end up infringing on privacy rights. They are especially prevalent among lower amount fines yet frequently correspond to serious legal infringements. This highlights the need for clearer regulatory standards and targeted educational initiatives to support smaller operators in understanding and meeting their GDPR obligations. Furthermore, the risk factors associated with surveillance technologies are increasingly relevant to other emerging systems, underscoring the importance of proactive, risk-based assessments in the design and deployment of new digital tools.

Taken together, these insights point to the value of an explainable, data-driven approach to understanding GDPR enforcement. By validating and extending prior findings, our XAI framework demonstrates the potential of computational methods to support legal interpretation while preserving transparency and accountability—critical features in both academic and policy contexts. Looking ahead, we plan to expand this work by incorporating more recent decisions and full-text analyses, thereby enhancing the granularity of linguistic and legal insights. Specifically, we will deploy a sentence-level topic model on the complete penalty texts to capture the logical-legal concatenation behind each fine and to reveal subtle divergences among Data Protection Authorities or industrial sectors. Furthermore, we will develop an automated, topic-by-topic summarization system that preserves the narrative structure of each decision and benchmark these machine-generated summaries against human-produced ones to ensure fidelity and usability In conclusion, the evolving landscape of GDPR enforcement cannot be fully understood through legal doctrine alone. It requires scalable analytic tools capable of capturing the multifaceted dynamics of regulation, compliance, and organizational behavior. This study contributes to that understanding by offering a replicable, interpretable framework that helps map the contours of data governance in a digital Europe.

## Uncited references

,

## Declaration of competing interest

The authors declare the following financial interests/personal relationships which may be considered as potential competing interests: Mario Santoro reports equipment, drugs, or supplies was provided by PiazzaCopernico srl. Albina Orlando reports financial support was provided by MUR - Italian Ministery for University and Research. If there are other authors, they declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgment

The SERICS project partially supported this research () under the MUR National Recovery and Resilience Plan funded by the European Union, NextGenerationEU.

## Data availability

Data will be made available on request.
