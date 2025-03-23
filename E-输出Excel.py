# 导入部分
# Group imports by logical modules
import openai
import os
from tabulate import tabulate
import pandas as pd
from datetime import datetime
from langchain_openai import ChatOpenAI


# Try to import openpyxl, which is needed for Excel writing
try:
    import openpyxl

    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Note: 'openpyxl' package not found. Install it using 'pip install openpyxl' to enable Excel export.")
    print("Results will be saved as CSV instead.")

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA


# LangChain integration modules
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
from langchain_community.callbacks.manager import get_openai_callback

#函数定义部分: 从用户那里获取PDF文件
def get_pdf_path():
    """Get PDF file path from user input"""
    pdf_path = input("Please enter the path to the PDF file: ")
    return pdf_path

# 使用PyMuPDFLoader加载PDF文件，将其转换为文档对象。
def load_pdf(pdf_path):
    """Load PDF file and return document object"""
    return PyMuPDFLoader(pdf_path).load()

# 文档的基本信息统计
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


# 用分割后的文档创建向量存储（使用OpenAI的嵌入模型），并显示令牌使用情况。
def format_environmental_results(result):
    """Format environmental analysis results as tables and return structured data"""
    # Extract the result string
    if isinstance(result, dict):
        # If it's a dictionary, try to get the result value
        environmental_data = result.get('result', str(result))
    else:
        environmental_data = result

    # Ensure it's a string
    environmental_data = str(environmental_data)

    # Lists to store extracted values
    extracted_values = []
    structured_data = []

    # Parse the data
    in_extracted_values = False
    missing_data_explanations = {}

    for line in environmental_data.split('\n'):
        line = line.strip()

        # Identify sections
        if "Extracted Environmental Metrics:" in line:
            in_extracted_values = True
            continue

        # Skip empty lines
        if not line:
            continue

        # Process key-value pairs
        if ':' in line and in_extracted_values:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Check if data is not found
            if "DATA NOT FOUND" in value:
                # Look for explanation in next lines
                explanation = ""
                found_explanation = False

                # Add to display list
                extracted_values.append([key, "DATA NOT FOUND"])

                # Add to structured data for Excel
                structured_data.append({
                    "Metric Name": key,
                    "Value": "DATA NOT FOUND",
                    "Page Number": "N/A",
                    "Comments": explanation if explanation else "No data available"
                })
                continue

            # Extract page number if present
            page_info = ""
            main_value = value
            page_number = "N/A"

            if '(' in value and ')' in value and 'page' in value.lower():
                main_value = value.split('(')[0].strip()
                page_info = value[value.find('('):].strip()
                # Extract just the page number
                if 'page' in page_info.lower():
                    page_parts = page_info.lower().replace('(', '').replace(')', '').split('page')
                    if len(page_parts) > 1:
                        page_number = page_parts[1].strip()

                value = f"{main_value} {page_info}"

            # Add to display list
            extracted_values.append([key, value])

            # Add to structured data for Excel
            structured_data.append({
                "Metric Name": key,
                "Value": main_value,
                "Page Number": page_number,
                "Comments": ""
            })

    # Print extracted values
    print("\n" + "=" * 80)
    print(f"{'Environmental Data Analysis Results':^76}")
    print("=" * 80)

    if extracted_values:
        print("\n" + "-" * 80)
        print(f"{'Extracted Environmental Metrics':^76}")
        print("-" * 80)
        print(tabulate(extracted_values, headers=["Metric Name", "Value"], tablefmt="grid"))

        # Count missing data
        missing_count = sum(1 for item in extracted_values if "DATA NOT FOUND" in item[1])
        if missing_count > 0:
            print(f"\nNOTE: {missing_count} out of {len(extracted_values)} metrics could not be found in the document.")
    else:
        print("\nNo environmental metrics found in the results.")

    print("=" * 80 + "\n")

    # Return structured data for Excel export
    return structured_data

# 将提取的环境数据保存为Excel文件
def save_to_excel(data, pdf_filename):
    """Save environmental data to Excel file or CSV if Excel is not available"""
    if not data:
        print("No data to export to file.")
        return

    # Create a DataFrame from the structured data
    df = pd.DataFrame(data)

    # Get the base name of the PDF file without extension
    base_name = os.path.splitext(os.path.basename(pdf_filename))[0]

    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if Excel export is available (openpyxl installed)
    if EXCEL_AVAILABLE:
        # Export to Excel
        excel_filename = f"{base_name}_environmental_metrics_{timestamp}.xlsx"

        # Create Excel writer
        writer = pd.ExcelWriter(excel_filename, engine='openpyxl')

        # Write DataFrame to Excel
        df.to_excel(writer, sheet_name='Environmental Metrics', index=False)

        # Auto-adjust columns' width
        for column in df:
            column_width = max(df[column].astype(str).map(len).max(), len(column))
            col_idx = df.columns.get_loc(column)
            writer.sheets['Environmental Metrics'].column_dimensions[chr(65 + col_idx)].width = column_width + 2

        # Save the Excel file
        writer.close()

        print(f"\nExcel report saved as: {excel_filename}")
    else:
        # Export to CSV as fallback
        csv_filename = f"{base_name}_environmental_metrics_{timestamp}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\nCSV report saved as: {csv_filename}")
        print("Install 'openpyxl' package using 'pip install openpyxl' to enable Excel export.")


def process_environmental_data_in_chunks(vectordb, llm, query_base, top_k=2):
    """Process environmental data by querying in smaller chunks to avoid context length issues"""
    # List of environmental metrics to extract 定义要提取的环境指标列表
    environmental_metrics = [
        "Total Global GHG emissions Scope 1, 2 & 3 (tonnes CO2e)",
        "Net Global GHG emissions Scope 1, 2 & 3 (tonnes CO2e)",
        "Total Premises Energy Use Consumed (MWh)",
        "Total Renewable energy consumption (MWh)",
        "Net energy consumption (MWh)",
        "Total Road transport energy use (MWh)",
        "Total Paper use (tonnes)",
        "Total Waste (tonnes)",
        "Waste Recycling Rate (%)",
        "Total Water consumption (m³)",
        "Total Sustainable Finance Deals ($m)",
        "Products for Personal and Business Customers ($m)"
    ]

    # Split metrics into groups of 4 to handle in separate queries
    metric_groups = [environmental_metrics[i:i + 4] for i in range(0, len(environmental_metrics), 4)]

    # Storage for all results
    all_results = []

    # Process each group
    for i, metric_group in enumerate(metric_groups):
        # Build focused query for this group of metrics
        metric_list = "\n".join([f"{j + 1}. {metric}" for j, metric in enumerate(metric_group)])

        query = f"""
        You are a senior environmental analyst specializing in corporate sustainability reporting. Your task is to carefully extract ONLY the following specific environmental data from the provided annual report and ensure all metrics are standardized with consistent units. Focus ONLY on the most recent fiscal year's data (current year).

        EXTRACTION GUIDELINES:
        1. Be thorough in searching for data across the entire document.
        2. Pay special attention to sections titled: "Environmental Performance", "Sustainability Report", "ESG Highlights", "Climate Change", "Carbon Footprint", "Energy Use", "Waste Management", "Water Management", or "Sustainable Finance".
        3. Recognize that environmental indicators may appear under various synonyms.
        4. STANDARDIZE ALL UNITS AS FOLLOWS:
           - GHG Emissions: Metric tonnes CO2e (CO2 equivalent)
           - Energy Use: MWh (Megawatt hours)
           - Paper Use: Metric tonnes
           - Waste: Metric tonnes
           - Water: Cubic meters (m³)
           - Financial Metrics: Millions of the report's currency (e.g., $m, €m, £m)

        5. Sentence IDENTIFICATION - EXTREMELY IMPORTANT:
           - For each value extracted, you MUST identify the EXACT Sentence where it appears

        6. If you find multiple values for the same indicator, prioritize:
           - Data explicitly labeled for the current year
           - Data from audited or verified sections
           - Absolute values rather than intensity metrics

        ENVIRONMENTAL METRICS TO EXTRACT (for current year only):
        {metric_list}

        OUTPUT FORMAT:
        Extracted Environmental Metrics:

        """

        # Add expected output format for each metric
        for metric in metric_group:
            metric_name = metric.split('(')[0].strip()
            query += f"{metric_name}: <value> (from sentence <sentence>)\n"

        # Find relevant documents for this set of metrics
        docs = vectordb.similarity_search(query, k=top_k)

        # Create QA chain for this specific query
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": top_k}),
            return_source_documents=True
        )

        print(f"\nProcessing metric group {i + 1}/{len(metric_groups)}")

        # Execute QA chain for this group
        with get_openai_callback() as cb:
            result = qa_chain.invoke(query)
            print(f"Token usage for group {i + 1}: {cb}")

            # Extract and store results
            if isinstance(result, dict):
                result_str = result.get('result', '')
            else:
                result_str = str(result)

            all_results.append(result_str)

    # Combine all results
    combined_result = "Extracted Environmental Metrics:\n\n"

    # Extract and format the metrics from individual results
    for result in all_results:
        if "Extracted Environmental Metrics:" in result:
            # Skip header line and extract actual metrics
            content = result.split("Extracted Environmental Metrics:", 1)[1].strip()
            combined_result += content + "\n\n"
        else:
            # Just add the result if it doesn't have the header
            combined_result += result.strip() + "\n\n"

    return combined_result

# 主函数: 加载PDF、处理文档、创建向量数据库、执行查询、格式化结果并保存。
def main():
    """Main function"""
    # Set OpenAI API key
    openai.api_key = " "

    # Initialize query history
    query_history = []

    # Vector storage parameters
    persist_directory = 'environmental_extraction_data'
    collection_name = 'environmental_extraction_index'

    # Step 1: Get PDF path and load
    print("\n" + "*" * 50)
    print(f"{'Environmental Data Analysis Tool':^46}")
    print("*" * 50)

    pdf_path = get_pdf_path()
    docs = load_pdf(pdf_path)
    print_document_info(docs)

    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)  # Reduced chunk size
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

    # Step 5: Create LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=openai.api_key)


    # Step 6: Base query template for environmental analysis (will be modified for each chunk)
    base_query = """
    You are a senior environmental analyst specializing in corporate sustainability reporting. Your task is to carefully extract environmental data from the provided annual report and ensure all metrics are standardized with consistent units. Focus ONLY on the most recent fiscal year's data (current year).
    """

    print("\n" + "-" * 50)
    print(f"{'Starting Environmental Data Analysis':^46}")
    print("-" * 50)

    # Step 7: Process the document in chunks to avoid context length issues
    result = process_environmental_data_in_chunks(vectordb, llm, base_query, top_k=2)

    # Step 8: Format results and save to Excel
    structured_data = format_environmental_results(result)

    # Save results to Excel file
    save_to_excel(structured_data, pdf_path)

    print("\n" + "-" * 50)
    print(f"{'Analysis Completed':^46}")
    print("-" * 50 + "\n")


if __name__ == "__main__":
    main()