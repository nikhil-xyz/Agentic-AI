import os
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo

import phi
from phi.playground import Playground, serve_playground_app
import uvicorn

load_dotenv()
phi.api = os.getenv("PHI_API_KEY")

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

app = Playground(agents=[web_search_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)