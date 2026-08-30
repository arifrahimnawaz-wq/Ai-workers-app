import os
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

st.set_page_config(page_title="AI Research & Writing Crew", page_icon="🧠", layout="centered")

st.title("🧠 AI Research & Writing Crew")
st.caption("Powered by CrewAI + Groq (Llama 3.1) — 100% free to run")

def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")

groq_api_key = get_groq_api_key()

if not groq_api_key:
    st.error(
        "GROQ_API_KEY not found. Add it to Streamlit Secrets "
        "(Settings → Secrets) or set it as an environment variable."
    )
    st.stop()

llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="groq/llama-3.1-70b-versatile",
    temperature=0.5,
)

_ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
search_tool = DuckDuckGoSearchRun(api_wrapper=_ddg_wrapper)

topic = st.text_input("Enter a topic for the blog post:", placeholder="e.g. The future of solar energy")
run_button = st.button("🚀 Generate Article", type="primary", use_container_width=True)

def build_crew(topic: str) -> Crew:
    researcher = Agent(
        role="Senior Research Analyst",
        goal=f"Find the most relevant, up-to-date, and accurate information about: {topic}",
        backstory=(
            "You are a meticulous research analyst with a knack for finding reliable, "
            "current information from the web and summarizing it clearly."
        ),
        tools=[search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Senior Content Writer",
        goal=f"Write an engaging, well-structured blog post about: {topic}",
        backstory=(
            "You are an experienced content writer who transforms raw research into "
            "clear, engaging, and well-organized articles with headings and a strong "
            "introduction and conclusion."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )

    research_task = Task(
        description=(
            f"Research the topic '{topic}' thoroughly using the web search tool. "
            "Gather key facts, recent developments, statistics, and expert perspectives. "
            "Summarize your findings in clear bullet points."
        ),
        expected_output="A structured summary of research findings with key facts and sources.",
        agent=researcher,
    )

    writing_task = Task(
        description=(
            f"Using the research findings, write a complete, well-structured blog post "
            f"about '{topic}'. Include an engaging title, an introduction, 3-5 body "
            "sections with subheadings, and a conclusion. Use Markdown formatting."
        ),
        expected_output="A complete, publish-ready blog post in Markdown format.",
        agent=writer,
        context=[research_task],
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, writing_task],
        process=Process.sequential,
        verbose=True,
    )

if run_button:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        with st.spinner("Researching and writing your article... this can take a minute."):
            try:
                crew = build_crew(topic.strip())
                result = crew.kickoff()

                st.success("Done! Here's your article:")
                st.markdown("---")
                st.markdown(str(result))

                st.download_button(
                    label="📥 Download as Markdown",
                    data=str(result),
                    file_name=f"{topic.strip().replace(' ', '_').lower()}.md",
                    mime="text/markdown",
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")

st.markdown("---")
st.caption(
    "Researcher agent uses DuckDuckGo search for live data. "
    "Writer agent compiles the research into a finished article. "
    "Both agents run on Groq's free Llama 3.1 70B model."
)
