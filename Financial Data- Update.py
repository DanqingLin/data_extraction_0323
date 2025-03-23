# Group imports by logical modules
import openai
from tabulate import tabulate

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader  #用于加载PDF文件
from langchain.text_splitter import RecursiveCharacterTextSplitter #将文档分成较小的块
from langchain.prompts import PromptTemplate #用于创建提示和问答链
from langchain.chains.question_answering import load_qa_chain #用于检索式问答
from langchain.chains import RetrievalQA #跟踪OpenAI API调用的统计信息
from langchain_openai import ChatOpenAI


# LangChain integration modules
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
from langchain_community.callbacks.manager import get_openai_callback


# 从用户获取PDF文件的路径
def get_pdf_path():
    """Get PDF file path from user input"""
    pdf_path = input("Please enter the path to the PDF file: ")
    return pdf_path


def load_pdf(pdf_path):
    """Load PDF file and return document object"""
    return PyMuPDFLoader(pdf_path).load()

# 打印文档的信息统计
def print_document_info(docs):
    """Print document information statistics"""
    print("\n" + "=" * 50)
    print(f"{'Document Information Statistics':^46}")
    print("=" * 50)
    print(f"Total document pages: {len(docs)}")
    print(f"Characters in first page: {len(docs[0].page_content)}")

    total = sum(len(doc.page_content) for doc in docs)
    print(f"Total characters in document: {total:,}")
    print("=" * 50 + "\n")


# 从分割的文档创建向量存储 》 使用OpenAI的嵌入 〉 指定集合名称和持久化目录 》 跟踪并打印令牌使用统计
def create_vector_store(split_docs, embeddings, collection_name, persist_directory):
    """Create and return vector store"""
    with get_openai_callback() as cb:
        vectorstore = Chroma.from_documents(
            split_docs,
            embeddings,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        print("\n" + "-" * 50)
        print(f"{'Vector Store Creation Completed':^46}")
        print("-" * 50)
        print(f"Token Usage Statistics:")
        print(cb)
        print("-" * 50 + "\n")

    return vectorstore


def format_financial_results(result):
    """Format financial analysis results as tables"""
    # Extract the result string
    if isinstance(result, dict):
        # If it's a dictionary, try to get the result value
        financial_data = result.get('result', str(result))
    else:
        financial_data = result

    # Ensure it's a string
    financial_data = str(financial_data)

    # Lists to store extracted values and ratios
    extracted_values = []
    computed_ratios = []

    # Parse the data
    in_extracted_values = False
    in_computed_ratios = False

    for line in financial_data.split('\n'):
        line = line.strip()

        # Identify sections
        if "Extracted Values:" in line:
            in_extracted_values = True
            in_computed_ratios = False
            continue
        elif "Computed Ratios:" in line:
            in_extracted_values = False
            in_computed_ratios = True
            continue

        # Skip empty lines
        if not line:
            continue

        # Process key-value pairs
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Extract source sentence if present
            sentence_info = ""
            if '(from' in value and ')' in value:
                main_value = value.split('(from')[0].strip()
                sentence_info = "(from" + value.split('(from')[1].strip()
                value = f"{main_value} {sentence_info}"

            # Add to appropriate list
            if in_extracted_values:
                extracted_values.append([key, value])
            elif in_computed_ratios:
                computed_ratios.append([key, value])

    # Print extracted values
    print("\n" + "=" * 80)
    print(f"{'Financial Data Analysis Results':^76}")
    print("=" * 80)

    if extracted_values:
        print("\n" + "-" * 80)
        print(f"{'Extracted Financial Indicators':^76}")
        print("-" * 80)
        print(tabulate(extracted_values, headers=["Indicator Name", "Value"], tablefmt="grid"))
    else:
        print("\nNo extracted values found in the results.")

    # Print calculated ratios
    if computed_ratios:
        print("\n" + "-" * 80)
        print(f"{'Calculated Financial Ratios':^76}")
        print("-" * 80)
        print(tabulate(computed_ratios, headers=["Ratio Name", "Value"], tablefmt="grid"))
    else:
        print("\nNo computed ratios found in the results.")

    print("=" * 80 + "\n")


# 主函数
def main():
    """Main function"""
    # Set OpenAI API key
    openai.api_key = " "

    # Initialize query history
    query_history = []

    # Vector storage parameters
    persist_directory = 'data_extraction_test'
    collection_name = 'data_extraction_test_index'

    # Step 1: Get PDF path and load
    print("\n" + "*" * 50)
    print(f"{'Financial Report Analysis Tool':^46}")
    print("*" * 50)

    pdf_path = get_pdf_path()
    docs = load_pdf(pdf_path)
    print_document_info(docs)

    # Step 2: Split documents 将文档分割成较小的块
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    split_docs = text_splitter.split_documents(docs)
    print(f"Document split into {len(split_docs)} chunks")

    # Step 3: Create embeddings and vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
    vectorstore = create_vector_store(split_docs, embeddings, collection_name, persist_directory)

    # Step 4: Load vector database
    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Step 5: Create LLM and QA chain
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai.api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # Step 6: Build financial analysis query and execute
    query = """
    You are a senior financial analyst with expertise in banking sector analysis. Your task is to carefully extract financial data from the provided bank's annual report and compute key financial ratios with precision. Use ONLY 2024 data for your extraction. Ignore the other years.

    EXTRACTION GUIDELINES:
    1. Be thorough in searching for data across the entire document.
    2. Pay special attention to sections titled: "Financial Highlights", "Key Financial Indicators", "Balance Sheet", "Income Statement", "Financial Review", or "Capital Management".
    3. Recognize that financial indicators may appear under various synonyms:
       - Net Income: Also "Profit", "Net Profit", "Net Earnings", "Profit After Tax"
       - Total Assets: Also "Assets", "Balance Sheet Total", "Total Balance Sheet"
       - Shareholders' Equity: Also "Total Equity", "Net Assets", "Book Value", "Total Shareholders' Funds"
       - Non-Performing Loans: Also "NPL", "Bad Debts", "Impaired Loans", "Defaulted Loans"
       - Total Loans: Also "Loans and Advances", "Credit Portfolio", "Loan Portfolio", "Net Loans"
       - Tier 1 Capital: Also "Core Capital", "Primary Capital", "CET1 Capital"
       - Tier 2 Capital: Also "Supplementary Capital", "Secondary Capital"
       - Risk-Weighted Assets: Also "RWA", "Risk-Adjusted Assets"

    4. ALWAYS convert all monetary values to millions (1,000,000) of the report's currency.
       - For example, if a value is given as "10,500,000" or "10.5 million", report as "10.5 million"
       - If a value is given as "1.2 billion", convert to "1,200 million"
       - Always include "million" after the value for clarity

    5. SOURCE SENTENCE IDENTIFICATION - EXTREMELY IMPORTANT:
       - For each value extracted, you MUST include the EXACT sentence where it appears
       - Extract the complete sentence containing the financial data
       - If the sentence is too long, include at least the relevant phrase with sufficient context
       - Record sentences in quotation marks
       - If a value appears multiple times, cite the clearest or most authoritative instance

    6. If you find multiple values for the same indicator, prioritize:
       - Data explicitly labeled for the current year
       - Data from consolidated financial statements
       - Data from the main table rather than footnotes
       - Absolute values rather than percentages

    REQUIRED FINANCIAL RATIOS:

    1. Return on Assets (ROA):
       - Formula: Net Income ÷ Total Assets
       - Express as percentage with 2 decimal places (e.g., "2.45%")

    2. Return on Equity (ROE):
       - Formula: Net Income ÷ Shareholders' Equity
       - Express as percentage with 2 decimal places (e.g., "12.75%")

    3. Non-Performing Loan (NPL) Ratio:
       - Formula: Non-Performing Loans ÷ Total Loans
       - Express as percentage with 2 decimal places (e.g., "3.21%")

    4. Capital Adequacy Ratio (CAR):
       - Formula: (Tier 1 Capital + Tier 2 Capital) ÷ Risk-Weighted Assets
       - Express as percentage with 2 decimal places (e.g., "15.62%")

    VALUES TO EXTRACT (for current year only):
    1. Net Income
    2. Total Assets
    3. Shareholders' Equity
    4. Non-Performing Loans
    5. Total Loans
    6. Tier 1 Capital
    7. Tier 2 Capital
    8. Risk-Weighted Assets

    OUTPUT FORMAT:
    Extracted Values:

    Net Income: <value in millions> million (from "<exact source sentence>")
    Total Assets: <value in millions> million (from "<exact source sentence>")
    Shareholders' Equity: <value in millions> million (from "<exact source sentence>")
    Non-Performing Loans: <value in millions> million (from "<exact source sentence>")
    Total Loans: <value in millions> million (from "<exact source sentence>")
    Tier 1 Capital: <value in millions> million (from "<exact source sentence>")
    Tier 2 Capital: <value in millions> million (from "<exact source sentence>")
    Risk-Weighted Assets: <value in millions> million (from "<exact source sentence>")

    Computed Ratios:

    ROA: <percentage with 2 decimal places>
    ROE: <percentage with 2 decimal places>
    NPL Ratio: <percentage with 2 decimal places>
    CAR: <percentage with 2 decimal places>

    Important: 
    1. If a value cannot be found after thorough examination, explicitly state "Value not found" in place of the value.
    2. If a ratio cannot be calculated due to missing data, write "Cannot compute, missing data" as the full result.
    3. If a ratio CAN be calculated, simply provide the percentage WITHOUT any explanatory text.
    4. Always convert and present all monetary values in millions, with "million" explicitly stated after each value.
    5. Always include the precise sentence for each extracted value.
    6. Double-check all extracted values for accuracy before calculating ratios.
    7. If a financial indicator is available in a summary table (e.g., statement of financial position), use that as the authoritative source. Avoid footnotes or narrative mentions unless no table data exists.
    """


    print("\n" + "-" * 50)
    print(f"{'Starting Financial Data Analysis':^46}")
    print("-" * 50)

    # Execute query
    docs = vectordb.similarity_search(query, k=5)
    print(f"Found {len(docs)} relevant document chunks")

    # Show document metadata only
    print("\nRelevant Document Metadata:")
    for i, doc in enumerate(docs, 1):
        print(f"Document {i}: {doc.metadata}")

    # Step 7: Execute QA chain and format results
    with get_openai_callback() as cb:
        result = qa_chain.invoke(query)
        format_financial_results(result)

        print("\n" + "-" * 50)
        print(f"{'API Usage Statistics':^46}")
        print("-" * 50)
        print(cb)
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()