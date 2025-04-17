#!/usr/bin/env python3
import argparse, re, csv, sys, os
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import Tool
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import ShellTool

class GrepRAGSystem:
    def __init__(self, file_path, chunk_size=1000, chunk_overlap=200, model="mistral:latest", show_work=True):
        self.file_path, self.chunk_size, self.chunk_overlap = file_path, chunk_size, chunk_overlap
        self.model, self.show_work, self.command_history = model, show_work, []
        try:
            self.llm = OllamaLLM(model=model)
        except Exception as e:
            print(f"Error initializing LLM: {str(e)}\nMake sure Ollama is running and the model is available")
            sys.exit(1)
        self.load_data()
        self.csv_structure = self.analyze_csv_structure()
        self.shell_tool = ShellTool(description="Execute shell commands. Use double quotes for arguments with spaces.")
        self.setup_agent()

    def setup_agent(self):
        tools = [self.shell_tool]
        react_prompt = PromptTemplate(
            input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
            template="""You are an assistant with access to shell commands.
Available tools: {tool_names}
{tools}
Use the following format:
Question: the input question for which you need to find information
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the command to run
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
Begin!
Question: {input}
{agent_scratchpad}"""
        )
        self.agent = create_react_agent(self.llm, tools, react_prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)

    def load_data(self):
        try:
            with open(self.file_path, 'r') as file:
                self.raw_data = file.read()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len)
            self.chunks = text_splitter.split_text(self.raw_data)
            print(f"Loaded {len(self.chunks)} chunks from {self.file_path}")
            with open(self.file_path, 'r') as file:
                reader = csv.reader(file)
                self.headers = next(reader)
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            sys.exit(1)

    def analyze_csv_structure(self):
        try:
            with open(self.file_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)
                sample_rows = []
                for _ in range(3):
                    try: sample_rows.append(next(reader))
                    except StopIteration: break
            structure = f"CSV Headers: {', '.join(header)}\n\nSample Data Format:\n"
            for i, row in enumerate(sample_rows):
                structure += f"Row {i + 1}: {', '.join(row)}\n"
            return structure
        except Exception as e:
            return f"Error analyzing CSV structure: {str(e)}"

    def grep(self, pattern, ignore_case=False, line_numbers=True, max_lines=None, context_lines=0, input_text=None):
        try:
            flags = re.IGNORECASE if ignore_case else 0
            regex = re.compile(pattern, flags)
            lines = input_text.splitlines() if input_text else open(self.file_path, 'r').readlines()
            results, match_count = [], 0
            for i, line in enumerate(lines, 1):
                is_match = regex.search(line) is not None
                if is_match:
                    match_count += 1
                    if context_lines > 0:
                        start = max(1, i - context_lines)
                        for j in range(start, i):
                            context_prefix = f"{j}: " if line_numbers else ""
                            results.append(f"{context_prefix}> {lines[j - 1].strip()}")
                    results.append(f"{i}: {line.strip()}" if line_numbers else line.strip())
                    if context_lines > 0:
                        end = min(len(lines), i + context_lines)
                        for j in range(i + 1, end + 1):
                            context_prefix = f"{j}: " if line_numbers else ""
                            results.append(f"{context_prefix}> {lines[j - 1].strip()}")
                    if max_lines and match_count >= max_lines:
                        results.append(f"... (stopped after {max_lines} matches)")
                        break
            return "\n".join(results) if results else f"No matches found for pattern '{pattern}'"
        except Exception as e:
            return f"Error performing grep: {str(e)}"

    def process_command(self, command, input_text=None):
        command = command.strip()
        cmd_parts = command.split(maxsplit=1)
        cmd_type = cmd_parts[0].lower() if cmd_parts else ""
        cmd_args = cmd_parts[1] if len(cmd_parts) > 1 else ""
        try:
            if input_text is None and not cmd_type.startswith("!"):
                self.command_history.append(command)
            command_handlers = {
                "!grep": lambda: self._handle_grep(cmd_args, input_text),
                "!agent": lambda: self._handle_agent(cmd_args, input_text),
                "!sort": lambda: self._handle_sort(cmd_args, input_text),
                "!head": lambda: self._handle_head(cmd_args, input_text),
                "!tail": lambda: self._handle_tail(cmd_args, input_text),
                "!count": lambda: self._handle_count(cmd_args, input_text),
                "!uniq": lambda: self._handle_uniq(cmd_args, input_text),
                "!shell": lambda: self._handle_shell(cmd_args, input_text),
                "!chunks": lambda: self._handle_chunks(cmd_args),
                "!model": lambda: self._handle_model(cmd_args),
                "!show": lambda: self._handle_show_work(cmd_args),
                "!history": lambda: "\n".join(self.command_history),
                "!grade": lambda: self._handle_grade(cmd_args),
                "!help": lambda: self._show_help(),
            }
            if cmd_type in command_handlers:
                return command_handlers[cmd_type]()
            else:
                return self.process_nl_query(command, input_text)
        except Exception as e:
            return f"Error processing command: {str(e)}"

    def _handle_grep(self, args, input_text=None):
        pattern_match = re.search(r'(?:"([^"]+)"|\'([^\']+)\'|(\S+))', args)
        if not pattern_match: return "Usage: !grep \"pattern\" [options]"
        pattern = next(g for g in pattern_match.groups() if g is not None)
        ignore_case = '-i' in args or '--ignore-case' in args
        line_numbers = not ('-n' in args or '--no-numbers' in args)
        max_lines = None
        max_match = re.search(r'-m\s+(\d+)|--max\s+(\d+)', args)
        if max_match:
            max_value = next(g for g in max_match.groups() if g is not None)
            max_lines = int(max_value)
        context_lines = 0
        context_match = re.search(r'-c\s+(\d+)|--context\s+(\d+)', args)
        if context_match:
            context_value = next(g for g in context_match.groups() if g is not None)
            context_lines = int(context_value)
        return self.grep(pattern, ignore_case, line_numbers, max_lines, context_lines, input_text)

    def _handle_agent(self, args, input_text=None):
        if not args: return "Usage: !agent your query here"
        return self.run_agent(args, input_text)

    def _handle_sort(self, args, input_text=None):
        reverse, numeric = "-r" in args, "-n" in args
        lines = input_text.splitlines() if input_text else open(self.file_path, 'r').readlines()
        if numeric:
            try:
                def extract_number(line):
                    matches = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                    return float(matches[0]) if matches else 0
                sorted_lines = sorted(lines, key=extract_number, reverse=reverse)
            except Exception:
                sorted_lines = sorted(lines, reverse=reverse)
        else:
            sorted_lines = sorted(lines, reverse=reverse)
        return "\n".join([line.strip() for line in sorted_lines])

    def _handle_head(self, args, input_text=None):
        match = re.search(r'-n\s*(\d+)', args)
        n = int(match.group(1)) if match else 10
        if input_text:
            lines = input_text.splitlines()[:n]
        else:
            with open(self.file_path, 'r') as file:
                lines = [file.readline() for _ in range(n)]
        return "\n".join([line.strip() for line in lines])

    def _handle_tail(self, args, input_text=None):
        match = re.search(r'-n\s*(\d+)', args)
        n = int(match.group(1)) if match else 10
        if input_text:
            lines = input_text.splitlines()[-n:]
        else:
            with open(self.file_path, 'r') as file:
                lines = file.readlines()[-n:]
        return "\n".join([line.strip() for line in lines])

    def _handle_count(self, args, input_text=None):
        text = input_text if input_text else open(self.file_path, 'r').read()
        count_lines = "-l" in args or not args
        count_words = "-w" in args or not args
        count_chars = "-c" in args or not args
        result = []
        if count_lines: result.append(f"Lines: {len(text.splitlines())}")
        if count_words: result.append(f"Words: {len(re.findall(r'\b\w+\b', text))}")
        if count_chars: result.append(f"Characters: {len(text)}")
        return "\n".join(result)

    def _handle_uniq(self, args, input_text=None):
        lines = input_text.splitlines() if input_text else open(self.file_path, 'r').readlines()
        count, prev_line, line_count, result = "-c" in args, None, 1, []
        for line in lines:
            line = line.strip()
            if line != prev_line:
                if prev_line is not None:
                    if count: result.append(f"{line_count} {prev_line}")
                    else: result.append(prev_line)
                prev_line, line_count = line, 1
            else:
                line_count += 1
        if prev_line is not None:
            if count: result.append(f"{line_count} {prev_line}")
            else: result.append(prev_line)
        return "\n".join(result)

    def _handle_shell(self, args, input_text=None):
        shell_command = re.search(r'"(.+)"', args)
        if shell_command:
            try:
                if input_text:
                    temp_file = "temp_pipe_input.txt"
                    with open(temp_file, 'w') as f: f.write(input_text)
                    command = f"{shell_command.group(1)} < {temp_file}" if "<" not in shell_command.group(1) else shell_command.group(1)
                else:
                    command = shell_command.group(1)
                result = self.shell_tool.run(command)
                if input_text and os.path.exists("temp_pipe_input.txt"): os.remove("temp_pipe_input.txt")
                return result
            except Exception as e:
                return f"Error executing shell command: {str(e)}"
        else:
            return "Usage: !shell \"your command here\""

    def _handle_chunks(self, args):
        try:
            size, overlap = args.split()
            return self.update_chunking(int(size), int(overlap))
        except ValueError:
            return "Invalid !chunks command. Use: !chunks size overlap"

    def _handle_model(self, args):
        try:
            self.model = args
            self.llm = OllamaLLM(model=args)
            self.setup_agent()
            return f"Model changed to {args}"
        except Exception as e:
            return f"Error changing model: {e}"

    def _handle_show_work(self, args):
        if args.lower() == "work":
            self.show_work = not self.show_work
            return f"Show work set to {self.show_work}"
        else:
            return "Usage: !show work"

    def _handle_grade(self, args):
        query_match = re.search(r'"(.+)"', args)
        if query_match:
            risky_patterns = [r'\b(exec|evaluate|run|execute)\b', r'\b(import|require|load)\b', r'\b(system|shell|os)\b', r'\b(inject|exploit|hack)\b']
            for pattern in risky_patterns:
                if re.search(pattern, query_match.group(1), re.IGNORECASE):
                    return f"Warning: Potential prompt injection detected in query: {query_match.group(1)}"
            return "No prompt injection risks detected."
        else:
            return "Usage: !grade \"your query here\""

    def _show_help(self):
        return """
✨ GREP-RAG COMMAND GUIDE ✨

📊 DATA ANALYSIS:
  > Your question in plain English
  > !grep "pattern" [-i] [-n] [-m N] [-c N]
    -i: ignore case, -n: no line #s, -m N: max matches, -c N: context lines

🔧 PIPE SYSTEM - Connect commands with |:
  > !grep "error" | !sort
  > What are the highest prices? | !sort -n -r | !head -n 5

📋 DATA COMMANDS:
  > !sort [-r] [-n]   : Sort lines (r:reverse, n:numeric)
  > !head [-n N]      : Show first N lines (default: 10)
  > !tail [-n N]      : Show last N lines (default: 10)
  > !count [-l/-w/-c] : Count lines/words/chars
  > !uniq [-c]        : Remove duplicate lines (c:show count)

🤖 ADVANCED:
  > !agent <command>  : Run agent with command
  > !shell "command"  : Execute shell command
  > !chunks size overlap : Change chunking parameters
  > !model modelname  : Change LLM model
  > !show work        : Toggle showing LLM work
  > !history          : Show command history
  > !grade "query"    : Grade query for injection risks
  > !help             : Show this guide
  > !quit             : Exit

TIP: Combine commands with | for powerful data processing pipelines!
"""

    def process_nl_query(self, query, input_text=None):
        if input_text is None: self.command_history.append(query)
        input_desc = "piped input data" if input_text else "the CSV file"
        grep_pattern_prompt = f"""
        Based on the user's query: "{query}"
        Generate 1-3 grep patterns that would help find relevant information in {input_desc}.
        {'' if input_text else f'The CSV file has the following structure:\n{self.csv_structure}'}
        For each pattern, explain why you chose it and what information it will help find.
        Format your patterns as: !grep "pattern" [options]
        Options:
        -i: ignore case
        -n: no line numbers
        -m <number>: maximum matches
        -c <number>: context lines
        """
        try:
            grep_patterns_response = self.llm.invoke(grep_pattern_prompt)
            patterns = re.findall(r'!grep\s+(?:"([^"]+)"|\'([^\']+)\')', grep_patterns_response)
            patterns = [p[0] if p[0] else p[1] for p in patterns]
            if not patterns:
                words = re.findall(r'\b\w+\b', query.lower())
                meaningful_words = [w for w in words if len(w) > 3 and w not in ['what', 'when', 'where', 'which', 'show', 'find', 'tell', 'give']]
                patterns = meaningful_words[:2] if meaningful_words else ['.']
            grep_results = []
            for pattern in patterns:
                result = self.grep(pattern, ignore_case=True, input_text=input_text)
                grep_results.append(f"Pattern: !grep \"{pattern}\" -i\n\nResults:\n{result}\n")
            data_source = "the piped input data" if input_text else "the CSV file"
            synthesis_prompt = f"""
            Based on the user's query: "{query}"
            I've searched {data_source} with the following grep results:
            {''.join(grep_results)}
            Please analyze these results and provide a comprehensive answer to the user's query.
            Reference specific data points from the grep results where relevant.
            """
            final_answer = self.llm.invoke(synthesis_prompt)
            if self.show_work:
                return f"""## Query: {query}

### Grep Patterns Used:
{grep_patterns_response}

### Search Results:
{''.join(grep_results)}

### Answer:
{final_answer}
"""
            else:
                return final_answer
        except Exception as e:
            return f"Error processing query: {str(e)}"

    def run_agent(self, query, input_text=None):
        try:
            if input_text:
                agent_input = {"input": f"{query}\n\nHere is the input data to work with:\n{input_text[:1000]}..." if len(input_text) > 1000 else f"{query}\n\nHere is the input data to work with:\n{input_text}"}
            else:
                agent_input = {"input": query}
            result = self.agent_executor.invoke(agent_input)
            return result.get("output", "No response from agent")
        except Exception as e:
            return f"Error running agent: {str(e)}"

    def update_chunking(self, new_size, new_overlap):
        self.chunk_size, self.chunk_overlap = new_size, new_overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len)
        self.chunks = text_splitter.split_text(self.raw_data)
        return f"Updated chunking parameters: size={new_size}, overlap={new_overlap}. New chunk count: {len(self.chunks)}"

    def process_piped_commands(self, command_string):
        self.command_history.append(command_string)
        commands = [cmd.strip() for cmd in command_string.split("|")]
        if not commands or all(not cmd for cmd in commands): return "Error: Empty command pipeline"
        if command_string.endswith("|"): return "Error: Incomplete pipe. Add a command after the | symbol."
        try:
            if not commands[0]: return "Error: Missing first command in pipeline"
            current_output = self.process_command(commands[0])
            progress = [f"✓ {commands[0]}"]
            for i, command in enumerate(commands[1:], 1):
                if not command.strip():
                    progress.append(f"⚠ Empty command at position {i+1}")
                    continue
                current_output = self.process_command(command, current_output)
                progress.append(f"✓ {command}")
            if len(commands) > 2 and self.show_work:
                pipeline_summary = " | ".join(progress)
                return f"Pipeline: {pipeline_summary}\n\nResult:\n{current_output}"
            else:
                return current_output
        except Exception as e:
            return f"Error in pipeline: {str(e)}"

    def process_chained_commands(self, command_string):
        self.command_history.append(command_string)
        parts, current_part, in_quotes, quote_char = [], "", False, None
        for char in command_string:
            if char in ('"', "'") and (not in_quotes or quote_char == char):
                in_quotes, quote_char = not in_quotes, char if not in_quotes else None
                current_part += char
            elif char in (';', '&') and not in_quotes:
                if current_part: parts.append(current_part.strip()); current_part = ""
                if char == '&' and len(command_string) > command_string.index(char) + 1:
                    if command_string[command_string.index(char) + 1] == '&':
                        parts.append('&&')
                        command_string = command_string[:command_string.index(char)] + ' ' + command_string[command_string.index(char) + 2:]
                    else: parts.append(';')
                else: parts.append(';')
            else: current_part += char
        if current_part: parts.append(current_part.strip())
        commands, operators = [], []
        for part in parts:
            if part in (';', '&&'): operators.append(part)
            else: commands.append(part)
        while len(operators) < len(commands) - 1: operators.append(';')
        results, stop_execution = [], False
        for i, cmd in enumerate(commands):
            if stop_execution:
                results.append(f"⚠️ Command skipped due to previous failure: {cmd}")
                continue
            if not cmd.strip(): continue
            if "|" in cmd: result = self.process_piped_commands(cmd)
            else: result = self.process_command(cmd)
            results.append(f"Command: {cmd}\nResult: {result}")
            if i < len(commands) - 1 and i < len(operators) and operators[i] == "&&" and "Error" in result:
                stop_execution = True
                results.append("⚠️ Stopping command chain due to error in previous command.")
        return "\n\n" + "\n\n".join(results)

def main():
    parser = argparse.ArgumentParser(description="Enhanced Grep-Prompted RAG System")
    parser.add_argument("--file", type=str, required=True, help="CSV file to analyze")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of each chunk in characters")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks in characters")
    parser.add_argument("--model", type=str, default="mistral:latest", help="Ollama model to use")
    parser.add_argument("--no-tutorial", action="store_true", help="Skip tutorial display at startup")
    parser.add_argument("--no-work", action="store_true", help="Don't show LLM work")
    args = parser.parse_args()

    rag_system = GrepRAGSystem(
        file_path=args.file,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        model=args.model,
        show_work=not args.no_work
    )

    if not args.no_tutorial:
        print("""
╔════════════════════════════════════════════════════════════════════╗
║                    📊 GREP-RAG SYSTEM QUICK START                   ║
╠════════════════════════════════════════════════════════════════════╣
║ • Ask questions in plain English: "What are the highest values?"   ║
║ • Use grep patterns: !grep "pattern" -i                            ║
║ • Use the pipe system: !grep "pattern" | !sort | !head -n 5        ║
║ • Try agents: !agent download and analyze the data                 ║
║                                                                    ║
║ Type !help for full command reference   |   !quit to exit          ║
╚════════════════════════════════════════════════════════════════════╝
        """)

    print(f"🚀 System ready! Analyzing: {args.file}")
    print("➡️  Enter your query or command below:")

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input: continue
            if user_input.lower() in ("!quit", "!exit"): break

            # Process command and print result
            if ";" in user_input or "&&" in user_input:
                result = rag_system.process_chained_commands(user_input)
            elif "|" in user_input:
                result = rag_system.process_piped_commands(user_input)
            else:
                result = rag_system.process_command(user_input)

            print(result)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()