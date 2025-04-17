#!/usr/bin/python3
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import textwrap

load_dotenv()

# Custom prompt template
CV_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""Answer the question based only on the following CV context.
    For questions about specific people, only use information from their CV.
    If you can't find the answer, say you don't know.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True},
    cache_folder="./embedding_cache"
)

vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def get_relevant_documents(query):
    """Improved document retrieval with name filtering"""
    # Check if query mentions a specific name
    name_matches = []
    for doc in vectorstore.docstore._dict.values():
        if "candidate_name" in doc.metadata:
            if doc.metadata["candidate_name"].lower() in query.lower():
                name_matches.append(doc.metadata["source"])
    
    if name_matches:
        # If name found, only search that person's documents
        return vectorstore.similarity_search(
            query,
            k=3,
            filter=lambda doc: doc.metadata["source"] in name_matches
        )
    else:
        # General search
        return vectorstore.similarity_search(query, k=3)

llm = ChatGroq(
    temperature=0.3,  # Lower temperature for more factual responses
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama3-70b-8192"
)

def ask_question(query):
    docs = get_relevant_documents(query)
    
    # Create context string
    context = "\n\n".join([
        f"Document {i+1} ({doc.metadata.get('source', 'Unknown')} - {doc.metadata.get('candidate_name', 'Unknown')}):\n{doc.page_content}"
        for i, doc in enumerate(docs)
    ])
    
    # Generate answer
    result = llm.invoke(CV_PROMPT.format(context=context, question=query))
    
    print("\nQuestion:", query)
    print("Answer:", textwrap.fill(result.content, width=80))
    
    print("\nSources:")
    seen_sources = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        if source not in seen_sources:
            print(f"- {source} ({doc.metadata.get('candidate_name', 'Unknown')})")
            seen_sources.add(source)

if __name__ == "__main__":
    print("CV Query System - Type 'quit' to exit")
    while True:
        query = input("\nEnter question: ").strip()
        if query.lower() == 'quit':
            break
        ask_question(query)






"""
Enter question: what do you know?

Question: what do you know?
Answer: I know about three individuals: Mark Gitonga, John, and Allan.   Mark Gitonga
has soft skills in communication, teamwork, problem-solving, time management,
emotional intelligence, and leadership. He has technical skills in network
designing and administration, penetration testing, cloud networking and
security, bash scripting, python programming, Linux commands, report writing,
and software installation and configuration. He has certifications in cloud and
network security, Cisco certified network associate (CCNA) certification, and
network security. He was an ICT support intern at the National Transport and
Safety Authority in Nairobi, Kenya from September 2022 to December 2022.  John
has technical skills in Python, Django, MySQL, JavaScript, AI and machine
learning, and Linux. He has soft skills in problem-solving, time management,
emotional intelligence, teamwork, and communication. He used technologies such
as Pycharm, Python, Django, MySQL, Git, and Mpesa API.  Allan is a technical
blog author who writes about topics such as Google Dorking and penetration
testing. He has technical skills in data analysis, AI, web development, network
design and administration, penetration testing, social engineering, reverse
engineering, and digital forensics. He is familiar with frameworks such as Power
Bi, MySQL, Excel, Flask, Security-Metaspoit, Maltego, Burp Suite, OWASP ZAP,
NMAP Scripting Engine, Hashcat, and John the Ripper. He can program in Python,
SQL, Shell Scripting (Python and Bash), C/C++, x86 Assembly, NIM, PowerShell,
Batch, HTML, CSS, and JavaScript.

Sources:
- mark.pdf (Unknown)
- john.pdf (Unknown)
- allan.pdf (Unknown)

Enter question: who is your prefered candidate for a data analyst role?

Question: who is your prefered candidate for a data analyst role?
Answer: Based on the provided CVs, I would say that the preferred candidate for a data
analyst role is the individual from Document 1 (allan.pdf - Unknown). This is
because they have explicitly mentioned "Data Analysis" as one of their technical
skills, and they also have experience with frameworks such as Power Bi, MySQL,
and Excel, which are commonly used in data analysis. Additionally, they have
mentioned "Python for Data Analysis" as one of their programming skills.

Sources:
- allan.pdf (Unknown)
- mark.pdf (Unknown)

Enter question: name the prefered candidate from the three in a role for web development?

Question: name the prefered candidate from the three in a role for web development?
Answer: Based on the provided CV context, I would recommend Allan (Document 1 and 2) as
the preferred candidate for a role in web development.   The reason for this is
that Allan has mentioned Web Development as one of their technical skills,
whereas John (Document 3) does not explicitly mention web development as a
skill.

Sources:
- allan.pdf (Unknown)
- john.pdf (Unknown)

Enter question: just mention the name between Allan, John and Mark who is the most prefered candidate  for a role in Backend development?

Question: just mention the name between Allan, John and Mark who is the most prefered candidate  for a role in Backend development?
Answer: Based on the CVs provided, I would say that John is the most preferred candidate
for a role in Backend development. This is because John has hands-on experience
with Django Framework, database modeling, and migrations using Django's ORM,
which are all relevant skills for backend development.

Sources:
- mark.pdf (Unknown)
- allan.pdf (Unknown)
- john.pdf (Unknown)

Enter question: between Allan, John and Mark who is the most prefered candidate for a role in Computer Networking?

Question: between Allan, John and Mark who is the most prefered candidate for a role in Computer Networking?
Answer: Based on the provided CVs, I would say that Mark is the most preferred candidate
for a role in Computer Networking.  Mark has a certification in Cisco Certified
Network Associate (CCNA) and has experience in Network Designing &
Administration, Penetration Testing, Cloud Networking and Security, and Network
Security. He also has experience in ICT Support Intern role where he carried out
network troubleshooting and installation and configuration of software.  Allan
has experience in IT SUPPORT, JOMO KENYATTA UNIVERSITY (ISS) where he performed
network administration tasks, but his experience is limited to a short period of
3 months. He also doesn't have any specific certifications in Computer
Networking.  There is no information about John, so I don't know anything about
his qualifications or experience.  Therefore, based on the available
information, Mark is the most preferred candidate for a role in Computer
Networking.

Sources:
- mark.pdf (Unknown)
- allan.pdf (Unknown)

Enter question: Mention job roles John can apply?

Question: Mention job roles John can apply?
Answer: Based on John's CV, it can be inferred that John can apply for job roles related
to:  1. Python Developer 2. Django Developer 3. MySQL Database Administrator 4.
AI/Machine Learning Engineer 5. Linux System Administrator  These job roles
align with John's technical skills mentioned in his CV.

Sources:
- mark.pdf (Unknown)
- allan.pdf (Unknown)
- john.pdf (Unknown)
"""