#!/usr/bin/env python3
"""
Conversation logging system for Talk2GPT-oss
Records all user queries, model responses, reasoning steps, and final answers
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import hashlib

class ConversationLogger:
    """Logger for recording complete conversation history with reasoning"""
    
    def __init__(self, log_directory: str = "conversation_logs"):
        """Initialize the conversation logger"""
        self.log_directory = log_directory
        self.ensure_log_directory()
        
        # Setup main conversation log file
        self.log_file = os.path.join(log_directory, "conversations.log")
        
        # Setup JSON log for structured data
        self.json_log_file = os.path.join(log_directory, "conversations.jsonl")
        
        # Setup Python logging
        self.setup_logging()
        
        # Session tracking
        self.session_id = None
        self.conversation_count = 0
    
    def ensure_log_directory(self):
        """Create log directory if it doesn't exist"""
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
            print(f"📁 Created conversation log directory: {self.log_directory}")
    
    def setup_logging(self):
        """Setup Python logging configuration"""
        # Create logger
        self.logger = logging.getLogger('conversation_logger')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def set_session_id(self, session_id: str):
        """Set the current session ID"""
        self.session_id = session_id
        self.conversation_count = 0
        self.logger.info(f"🔄 New session started: {session_id}")
    
    def generate_conversation_id(self, user_query: str) -> str:
        """Generate a unique conversation ID"""
        timestamp = datetime.now().isoformat()
        query_hash = hashlib.md5(user_query.encode()).hexdigest()[:8]
        return f"{timestamp}_{query_hash}"
    
    def log_conversation(self, 
                        user_query: str, 
                        raw_response: str, 
                        cleaned_response: str, 
                        reasoning_content: Optional[str] = None,
                        reasoning_detected: bool = False,
                        response_time_ms: Optional[float] = None,
                        model_settings: Optional[Dict[str, Any]] = None):
        """
        Log a complete conversation exchange
        
        Args:
            user_query: The user's original question
            raw_response: The raw model response before cleaning
            cleaned_response: The final cleaned response shown to user
            reasoning_content: Extracted reasoning content (if any)
            reasoning_detected: Whether reasoning was detected
            response_time_ms: Response generation time in milliseconds
            model_settings: Model parameters used (max_tokens, temperature, etc.)
        """
        self.conversation_count += 1
        conversation_id = self.generate_conversation_id(user_query)
        
        # Create conversation entry
        conversation_entry = {
            "conversation_id": conversation_id,
            "session_id": self.session_id,
            "conversation_number": self.conversation_count,
            "timestamp": datetime.now().isoformat(),
            "user_query": user_query,
            "model_response": {
                "raw_response": raw_response,
                "cleaned_response": cleaned_response,
                "reasoning_detected": reasoning_detected,
                "reasoning_content": reasoning_content
            },
            "metadata": {
                "response_time_ms": response_time_ms,
                "model_settings": model_settings,
                "has_reasoning": bool(reasoning_content and reasoning_content.strip()),
                "response_length": len(raw_response),
                "cleaned_response_length": len(cleaned_response)
            }
        }
        
        # Log to text file (human readable)
        self._log_text_format(conversation_entry)
        
        # Log to JSON file (machine readable)
        self._log_json_format(conversation_entry)
        
        # Create individual conversation file for easy access
        self._create_individual_log(conversation_entry)
        
        return conversation_id
    
    def _log_text_format(self, entry: Dict[str, Any]):
        """Log conversation in human-readable text format"""
        separator = "=" * 80
        
        log_text = f"""
{separator}
CONVERSATION #{entry['conversation_number']} - {entry['timestamp']}
Session: {entry['session_id']} | ID: {entry['conversation_id']}
{separator}

👤 USER QUERY:
{entry['user_query']}

🤖 MODEL RESPONSE (RAW):
{entry['model_response']['raw_response']}

✨ FINAL ANSWER (CLEANED):
{entry['model_response']['cleaned_response']}
"""
        
        if entry['model_response']['reasoning_content']:
            log_text += f"""
🧠 REASONING STEPS:
{entry['model_response']['reasoning_content']}
"""
        
        metadata = entry['metadata']
        log_text += f"""
📊 METADATA:
- Reasoning Detected: {entry['model_response']['reasoning_detected']}
- Has Reasoning Content: {metadata['has_reasoning']}
- Response Time: {metadata['response_time_ms']}ms
- Raw Response Length: {metadata['response_length']} chars
- Cleaned Response Length: {metadata['cleaned_response_length']} chars
- Model Settings: {metadata['model_settings']}

{separator}
"""
        
        self.logger.info(log_text)
    
    def _log_json_format(self, entry: Dict[str, Any]):
        """Log conversation in JSON Lines format for machine processing"""
        try:
            with open(self.json_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️ Error writing JSON log: {e}")
    
    def _create_individual_log(self, entry: Dict[str, Any]):
        """Create individual conversation file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}_{entry['conversation_number']}.txt"
            filepath = os.path.join(self.log_directory, "individual", filename)
            
            # Create individual directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            content = f"""Talk2GPT-oss Conversation Log
Generated: {entry['timestamp']}
Session: {entry['session_id']}
Conversation: #{entry['conversation_number']}

USER QUERY:
{entry['user_query']}

MODEL RESPONSE (Raw):
{entry['model_response']['raw_response']}

FINAL ANSWER (Cleaned):
{entry['model_response']['cleaned_response']}
"""
            
            if entry['model_response']['reasoning_content']:
                content += f"""
REASONING PROCESS:
{entry['model_response']['reasoning_content']}
"""
            
            content += f"""
TECHNICAL DETAILS:
- Reasoning Detected: {entry['model_response']['reasoning_detected']}
- Response Time: {entry['metadata']['response_time_ms']}ms
- Model Settings: {entry['metadata']['model_settings']}
"""
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            print(f"⚠️ Error creating individual log: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            "session_id": self.session_id,
            "conversation_count": self.conversation_count,
            "log_directory": self.log_directory,
            "log_files": {
                "main_log": self.log_file,
                "json_log": self.json_log_file
            }
        }
    
    def search_conversations(self, query: str, limit: int = 10) -> list:
        """Search through conversation history"""
        results = []
        try:
            if os.path.exists(self.json_log_file):
                with open(self.json_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if (query.lower() in entry['user_query'].lower() or 
                                query.lower() in entry['model_response']['cleaned_response'].lower()):
                                results.append(entry)
                                if len(results) >= limit:
                                    break
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"⚠️ Error searching conversations: {e}")
        
        return results
    
    def export_conversations(self, format: str = "txt", date_filter: Optional[str] = None) -> str:
        """Export conversations in specified format"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "txt":
            export_file = os.path.join(self.log_directory, f"export_{timestamp}.txt")
            # Copy the main log file
            import shutil
            if os.path.exists(self.log_file):
                shutil.copy2(self.log_file, export_file)
                return export_file
        
        elif format == "json":
            export_file = os.path.join(self.log_directory, f"export_{timestamp}.json")
            conversations = []
            
            try:
                if os.path.exists(self.json_log_file):
                    with open(self.json_log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                if not date_filter or date_filter in entry['timestamp']:
                                    conversations.append(entry)
                            except json.JSONDecodeError:
                                continue
                
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(conversations, f, ensure_ascii=False, indent=2)
                
                return export_file
                
            except Exception as e:
                print(f"⚠️ Error exporting to JSON: {e}")
                return None
        
        return None


# Global logger instance
conversation_logger = ConversationLogger()


def initialize_conversation_logger(session_id: str):
    """Initialize the conversation logger with session ID"""
    conversation_logger.set_session_id(session_id)
    return conversation_logger


def log_conversation_exchange(user_query: str, 
                            raw_response: str, 
                            cleaned_response: str, 
                            reasoning_content: Optional[str] = None,
                            reasoning_detected: bool = False,
                            response_time_ms: Optional[float] = None,
                            model_settings: Optional[Dict[str, Any]] = None):
    """Convenience function to log a conversation exchange"""
    return conversation_logger.log_conversation(
        user_query=user_query,
        raw_response=raw_response,
        cleaned_response=cleaned_response,
        reasoning_content=reasoning_content,
        reasoning_detected=reasoning_detected,
        response_time_ms=response_time_ms,
        model_settings=model_settings
    )


if __name__ == "__main__":
    # Test the logger
    logger = ConversationLogger()
    logger.set_session_id("test_session_001")
    
    # Test conversation
    test_user_query = "What is 2 + 2?"
    test_raw_response = "analysis Let me calculate this simple math problem. 2 + 2 = 4. assistantfinal 4"
    test_cleaned_response = "4"
    test_reasoning = "analysis Let me calculate this simple math problem. 2 + 2 = 4. assistantfinal"
    
    conversation_id = logger.log_conversation(
        user_query=test_user_query,
        raw_response=test_raw_response,
        cleaned_response=test_cleaned_response,
        reasoning_content=test_reasoning,
        reasoning_detected=True,
        response_time_ms=1250.5,
        model_settings={"max_tokens": 512, "temperature": 0.7}
    )
    
    print(f"✅ Test conversation logged with ID: {conversation_id}")
    print(f"📁 Log directory: {logger.log_directory}")
