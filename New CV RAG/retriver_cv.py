#!/usr/bin/python3
import os
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_chroma import Chroma

#define persistent directory
current_dir=os.path.dirname(os.path.abspath(__file__))
db_dir=os.path.join(current_dir,"db")
persistent_directory=os.path.join(db_dir, "chroma_db_with_metadata")

#define the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},  # Specify device
    encode_kwargs={'normalize_embeddings': True}  # Helps with similarity
)

#load the existing vector store
db=Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

#define the users question
query="allan email address?"

#Retrieve relevant documents based on the query
retriever = db.as_retriever(
    search_type="similarity",  # Changed to basic similarity search
    search_kwargs={"k": 5}     # Removed score_threshold parameter
)

relevant_docs=retriever.invoke(query)

#display the relevant results with metadata
print("\n---Relevant Documents----")
for i, doc in enumerate(relevant_docs,1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source:{doc.metadata['source']}\n")


"""
 python .\retriver_cv.py

---Relevant Documents----
Document 1:
Programming:  Python for Data Analysis, SQL, Shell Scripting (Python & Bash), and C/C++, x 86 Assembly, NIM,
PowerShell, Batch, HTML, CSS and Java Script.
Soft skill:  Time management, customer service, working under pressure, and group work.

Source:allan.pdf

Document 2:
dates and fines to extended days.
Technologies used: Pycharm, Python, Django, Mysql, Git, Mpesa API
SKILLS AND TECHNOLOGIES
Technical skills: Python, Django, MySQL, JavaScript, AI & Machine learnin, Linux
Soft skills: Problem-solving, Time Management, Emotional Intelligence, team work,
Communication.

Source:john.pdf

Document 3:
platforms like GitHub and Discord for C2 operations.
Technical Blog Author    link: https://medium.com/@karisallan237
I started writing on Medium, with my first article focusing on Google Dorking—how hackers use it to gather information for 
penetration testing. New articles have been dropped recently such as turning NMAP to a full vulnerability scanner.

SKILLS AND TECHNOLOGY
Technical skills: Data Analysis, AI(Prompt-Engineering, Langchain, RAG ), Web Development, Network Design and
Administration, Penetration Testing, Social Engineering, Reverse Engineering and Digital Forensics.
Frameworks: Data Analysis (Power Bi, MySQL, Excel), Flask, Security-Metaspoit, Maltego, Burp Suite, OWASP ZAP,
NMAP Scripting Engine, Hashcat, and John the Ripper.
Programming:  Python for Data Analysis, SQL, Shell Scripting (Python & Bash), and C/C++, x 86 Assembly, NIM,
PowerShell, Batch, HTML, CSS and Java Script.

Source:allan.pdf

Document 4:
minded individuals to achieve goals.

Professional Experience
ICT Support Intern    September 2022 - December 2022
National Transport and Safety Authority, Nairobi, Kenya
While in National Transport and Safety Authority I undertook the following:
• Carrying out first level user support to staff.
• Offered Preventive maintenance of laptops and PCs in Headquarters.
• Provided first level support of network troubleshooting, on laptops,
desktops and printers.
• Installation and configuration of software such as windows 10 and 11,
Checkpoint Antivirus, Office 365, Fortinet VPN and Dynamic ERP.
• Assisted in creation and management of users in Kyocera printing
management system.
Education
Jan 2024 – August 2024
Cloud and Network Security, CyberShujaa, Nairobi Kenya.

Sep 2023 - Mar 2024
Cyber Security, Coursera Google, Nairobi Kenya.

Jan 2023 - Jul 2023
Cisco Certified Network Associate
Jomo Kenyatta University of Agriculture and Technology, Nairobi, Kenya.

Source:mark.pdf

Document 5:
MARK GITONGA
CONTACT
 +254716133838
 mrkgitonga@gmail.com
 linkedin.com/in/MarkGitonga
Soft Skill
 Communication
 Teamwork
 Problem-Solving
 Time Management
 Emotional Intelligence
 Leadership
Technical Skill
Network Designing & Administration
Penetration Testing
Cloud Networking and Security
Bash Scripting
Python Programming
Linux Commands
Report Writting
Software Installation and
Configuration

CERTIFICATIONS
Cloud and Network Security
Cisco Certified Network Associate
(CCNA) Certification
Network Security
Google Cyber Security
Kenya Rollball Federation

I am a Dedicated technology enthusiast with experience in achieving tangible
results and cross-team collaboration. Proactive and excited to partner with like-
minded individuals to achieve goals.

Professional Experience
ICT Support Intern    September 2022 - December 2022
National Transport and Safety Authority, Nairobi, Kenya

Source:mark.pdf

"""