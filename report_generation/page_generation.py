import os
import webbrowser
import shutil
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import socket
from jinja2 import Environment, FileSystemLoader
from pydantic import BaseModel, Field
from typing import List

from agents.recommendation_agent import RecommendationAgent

from agents.llm_tools import safe_arxiv_retrieve_tool, safe_cve_search_tool

from report_generation.parse_results import (
    parse_arxiv_results,
    parse_classifier_results,
    parse_cve_results,
    parse_product_results,
)

from agents.llm_tools import CVE
from agents.recommendation_agent import SecurityProduct


class ArxivPaper(BaseModel):
    title: str = Field(description="Title of the paper")
    authors: str = Field(description="Authors of the paper")
    summary: str = Field(description="Summary/abstract of the paper")
    url: str = Field(description="URL to the paper on arXiv")
    published: str = Field(description="Publication date")


class SecurityAnalysis(BaseModel):
    cves: List[CVE] = Field(description="List of related CVE vulnerabilities")
    products: List[SecurityProduct] = Field(
        description="List of recommended security products"
    )
    research: List[ArxivPaper] = Field(description="List of relevant research papers")


class ReportPageGeneration:
    def __init__(self):
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.templates_dir = self.base_dir / "templates"
        self.reports_dir = self.base_dir / "reports"
        self.static_dir = self.reports_dir / "static"

        self.reports_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)

        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)), autoescape=True
        )

        def format_datetime(value):
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%d %H:%M:%S UTC")
            return value

        self.jinja_env.filters["format_datetime"] = format_datetime

        self.ra = RecommendationAgent()

    def __copy_static_files(self):
        tailwind_css = """
        @import 'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css';
        @import 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css';
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        body {
            font-family: 'Inter', sans-serif;
        }
        """

        css_file = self.static_dir / "style.css"
        css_file.write_text(tailwind_css)

    def generate_report(self, samples_df, classifier_prediction):
        def generate_sample_report(sample_info):
            template = self.jinja_env.get_template("traffic_analysis.html")
            sample_info["current_time"] = datetime.utcnow()
            output = template.render(**sample_info)

            output_path = self.reports_dir / sample_info["detail_url"]
            output_path.write_text(output)

        def generate_index_page(all_samples):
            template = self.jinja_env.get_template("index.html")
            output = template.render(
                samples=all_samples, current_time=datetime.utcnow()
            )

            output_path = self.reports_dir / "index.html"
            output_path.write_text(output)

        if self.reports_dir.exists():
            shutil.rmtree(self.reports_dir)
        self.reports_dir.mkdir()
        self.static_dir.mkdir()

        self.__copy_static_files()

        all_samples = []

        for i, sample in enumerate(samples_df.iterrows(), 1):
            current_sample = sample[1].to_frame().T

            sample_info = {
                "id": i,
                "dest_ip": sample[1]["Dst IP"],
                "protocol": int(sample[1]["Protocol"]),
                "flow_bytes_s": f"{sample[1]['Flow Bytes/s']:.2f}",
                "source_ip": sample[1]["Src IP"],
                "dest_port": int(sample[1]["Dst Port"]),
                "flow_duration": f"{sample[1]['Flow Duration']:.2f}",
                "flow_packets_s": f"{sample[1]['Flow Packets/s']:.2f}",
                "source_port": int(sample[1]["Src Port"]),
                "detail_url": f"sample_{i}.html",
            }

            sample_info.update(parse_classifier_results(classifier_prediction))

            # traffic_type = sample_info["traffic_type"] -- During inference, there is no "Label" column
            if isinstance(classifier_prediction, dict):
                traffic_type = classifier_prediction["final_prediction"]
            else:
                traffic_type = classifier_prediction
            try:
                if traffic_type.lower() == "icmp flood":
                    query = 'cat:cs.CR AND (ti:"DDoS" OR ti:"denial of service" OR abs:"ICMP flood")'
                elif traffic_type.lower() == "benign":
                    query = 'cat:cs.CR AND (ti:"network traffic classification" OR ti:"intrusion detection")'
                else:
                    query = (
                        f'cat:cs.CR AND (ti:"{traffic_type}" OR abs:"{traffic_type}")'
                    )

                print(f"\nSearching ArXiv with query: {query}")
                try:
                    # Use direct ArxivRetriever to get document objects instead of string results
                    from langchain_community.retrievers import ArxivRetriever

                    arxiv_retriever = ArxivRetriever(
                        load_max_docs=5,
                        doc_content_chars_max=None,
                        max_retries=3,
                    )
                    arxiv_results = arxiv_retriever.get_relevant_documents(query)
                    print(f"Found {len(arxiv_results)} papers")

                    if not arxiv_results:
                        fallback_query = 'cat:cs.CR AND (ti:"network security" OR ti:"intrusion detection")'
                        print(f"\nNo results, trying fallback query: {fallback_query}")
                        arxiv_results = arxiv_retriever.get_relevant_documents(
                            fallback_query
                        )
                        print(f"Found {len(arxiv_results)} papers with fallback query")
                except Exception as e:
                    print(f"Error in ArXiv search: {str(e)}")
                    arxiv_results = []

                sample_info["arxiv_articles"] = parse_arxiv_results(arxiv_results)
                print(
                    f"Total papers found and parsed: {len(sample_info['arxiv_articles'])}"
                )

            except Exception as e:
                print(f"Error retrieving research papers: {str(e)}")
                sample_info["arxiv_articles"] = []

            cve_results = safe_cve_search_tool(traffic_type)
            sample_info["cves"] = parse_cve_results(cve_results)

            try:
                if traffic_type.lower() == "icmp flood":
                    product_query = "DDoS and ICMP flood protection"
                elif traffic_type.lower() == "benign":
                    product_query = "network monitoring and intrusion detection"
                else:
                    product_query = f"{traffic_type} attack prevention"

                print(f"\nGetting product recommendations for: {product_query}")
                products = self.ra.get_llm_product_recommendation(product_query)
                sample_info["products"] = products
                print(f"Found {len(products)} product recommendations")

            except Exception as e:
                print(f"Error getting product recommendations: {str(e)}")
                sample_info["products"] = []

            all_samples.append(sample_info)

            generate_sample_report(sample_info)

        generate_index_page(all_samples)

        return str(self.reports_dir)

    def serve_reports(self):
        def find_available_port(start_port=8000):
            port = start_port
            while port < start_port + 100:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("", port))
                        return port
                except OSError:
                    port += 1
            raise RuntimeError("Could not find an available port")

        def start_http_server(port):
            os.chdir(str(self.reports_dir.absolute()))

            server = HTTPServer(("", port), SimpleHTTPRequestHandler)
            server_thread = threading.Thread(target=server.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            return server

        port = find_available_port()

        print(f"\nServing reports from: {self.reports_dir.absolute()}")
        print(f"Contents of reports directory:")
        for item in self.reports_dir.iterdir():
            print(f"  - {item.name}")

        server = start_http_server(port)

        url = f"http://0.0.0.0:{port}/index.html"
        print(f"\nStarting report server at {url}")
        webbrowser.open(url)

        try:
            print("\nPress Ctrl+C to stop the server...")
            while True:
                pass
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()
            server.server_close()
