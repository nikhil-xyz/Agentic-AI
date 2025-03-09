import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

load_dotenv()

web_search_agent = Agent(
    name = "Web Search Agent",
    # model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    model=Groq(id="llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions = ["Always include sources"],
    markdown = True,
    show_tool_calls = True
)

finance_agent = Agent(
    # model = Groq(id = "llama-3.3-70b-versatile", api_key=groq_api_key),
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls = True,
    markdown = True,
    instructions = ["use tables to display data"],
    debug_mode = True
)

agent_team = Agent(
    # model=Groq(id="llama-3.3-70b-versatile", api_key=groq_api_key),
    model=Groq(id="llama-3.3-70b-versatile"),
    team = [web_search_agent, finance_agent],
    markdown = True,
    show_tool_calls = True,
    instructions = ["Always include sources", "Use tables to display data"]
)

agent_team.print_response("summarize and compare analyst recommandations and fundamentals for TESLA and NVDA", stream=True)