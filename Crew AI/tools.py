#!/usr/bin/python3
import os
from exa_py import Exa
from langchain.agents import tool
from langchain.tools import Tool

class ExaSearchToolset():
    @tool
    def search(query: str):
        """Search for a webpage based on the query"""
        return ExaSearchToolset._exa().search(f"{query}", use_autoprompt=True, num_results=3)
    
    @tool
    def find_similar(url: str):
        """Searches for a webpage similar to the given URL"""
        return ExaSearchToolset._exa().find_similar(url, num_results=3)
    
    @tool
    def get_contents(ids: str):
        """Get the contents of a web page"""
        ids = eval(ids)
        contents = str(ExaSearchToolset._exa().get_contents(ids))
        contents = contents.split("URL")
        contents = [content[:1000] for content in contents]
        return "\n\n".join(contents)
    
    def tools():
        return [
            Tool.from_function(
                func=ExaSearchToolset.search,
                name="ExaSearch",
                description="Search for a webpage based on the query"
            ),
            Tool.from_function(
                func=ExaSearchToolset.find_similar,
                name="ExaFindSimilar",
                description="Find similar web pages to the given URL"
            ),
            Tool.from_function(
                func=ExaSearchToolset.get_contents,
                name="ExaGetContents",
                description="Get the contents of a webpage"
            )
        ]

    def _exa():
        return Exa(api_key=os.environ.get('EXA_API_KEY'))