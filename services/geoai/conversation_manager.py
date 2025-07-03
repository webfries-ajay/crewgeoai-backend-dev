import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI

class ProjectChatHistory(BaseChatMessageHistory):
    """Custom chat history implementation for project-based conversations"""
    
    def __init__(self, project_id: str, user_id: str):
        self.project_id = project_id
        self.user_id = user_id
        self.messages: List[BaseMessage] = []
        self._setup_storage()
        self._load_history()
    
    def _setup_storage(self):
        """Setup storage directory for conversation history"""
        self.storage_dir = Path("conversation_history") / self.user_id / self.project_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_dir / "chat_history.json"
    
    def _load_history(self):
        """Load conversation history from storage"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for msg_data in data.get('messages', []):
                        if msg_data['type'] == 'human':
                            self.messages.append(HumanMessage(content=msg_data['content']))
                        elif msg_data['type'] == 'ai':
                            self.messages.append(AIMessage(content=msg_data['content']))
                        elif msg_data['type'] == 'system':
                            self.messages.append(SystemMessage(content=msg_data['content']))
            except Exception as e:
                print(f"Error loading chat history: {e}")
                self.messages = []
    
    def _save_history(self):
        """Save conversation history to storage"""
        try:
            history_data = {
                'project_id': self.project_id,
                'user_id': self.user_id,
                'last_updated': datetime.now().isoformat(),
                'messages': []
            }
            
            for msg in self.messages:
                if isinstance(msg, HumanMessage):
                    msg_type = 'human'
                elif isinstance(msg, AIMessage):
                    msg_type = 'ai'
                elif isinstance(msg, SystemMessage):
                    msg_type = 'system'
                else:
                    continue
                
                history_data['messages'].append({
                    'type': msg_type,
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat()
                })
            
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving chat history: {e}")
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history"""
        self.messages.append(message)
        self._save_history()
    
    def add_user_message(self, message: str) -> None:
        """Add a user message to the history"""
        self.add_message(HumanMessage(content=message))
    
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the history"""
        self.add_message(AIMessage(content=message))
    
    def clear(self) -> None:
        """Clear the conversation history"""
        self.messages = []
        if self.history_file.exists():
            self.history_file.unlink()
    
    def get_messages(self) -> List[BaseMessage]:
        """Get all messages in the conversation"""
        return self.messages.copy()

class ConversationManager:
    """Manages conversations with persistent history and intelligent memory management"""
    
    def __init__(self):
        self.conversations: Dict[str, Dict[str, Any]] = {}
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize the LLM for conversation summary if needed"""
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.llm = ChatOpenAI(
                    api_key=api_key,
                    model=os.getenv('OPENAI_MODEL', "gpt-4o"),
                    temperature=0.1
                )
        except Exception as e:
            print(f"Warning: Could not initialize LLM for conversation summary: {e}")
    
    def get_conversation_key(self, project_id: str, user_id: str) -> str:
        """Generate a unique key for the conversation"""
        return f"{user_id}:{project_id}"
    
    def get_or_create_conversation(self, project_id: str, user_id: str, master_prompt: str = None) -> Dict[str, Any]:
        """Get existing conversation or create a new one with persistent history"""
        conv_key = self.get_conversation_key(project_id, user_id)
        
        if conv_key not in self.conversations:
            # Create new conversation with persistent history
            chat_history = ProjectChatHistory(project_id, user_id)
            
            # Create conversation summary buffer memory for intelligent memory management
            memory = None
            if self.llm:
                try:
                    memory = ConversationSummaryBufferMemory(
                        llm=self.llm,
                        chat_memory=chat_history,
                        max_token_limit=2000,  # Keep last 2000 tokens of conversation
                        return_messages=True,
                        memory_key="chat_history"
                    )
                except Exception as e:
                    print(f"Warning: Could not create summary memory, using basic memory: {e}")
            
            # If summary memory failed, use basic memory
            if not memory:
                from langchain.memory import ConversationBufferMemory
                memory = ConversationBufferMemory(
                    chat_memory=chat_history,
                    return_messages=True,
                    memory_key="chat_history"
                )
            
            self.conversations[conv_key] = {
                'project_id': project_id,
                'user_id': user_id,
                'chat_history': chat_history,
                'memory': memory,
                'master_prompt': master_prompt,
                'created_at': datetime.now(),
                'last_activity': datetime.now()
            }
            
            print(f"🆕 Created new conversation for project {project_id}, user {user_id}")
            print(f"📚 Loaded {len(chat_history.get_messages())} previous messages")
        else:
            # Update last activity and master prompt if provided
            self.conversations[conv_key]['last_activity'] = datetime.now()
            if master_prompt:
                self.conversations[conv_key]['master_prompt'] = master_prompt
        
        return self.conversations[conv_key]
    
    def add_user_message(self, project_id: str, user_id: str, message: str):
        """Add a user message to the conversation"""
        conversation = self.get_or_create_conversation(project_id, user_id)
        conversation['chat_history'].add_user_message(message)
        conversation['last_activity'] = datetime.now()
        
        print(f"💬 Added user message to conversation {project_id}")
    
    def add_ai_message(self, project_id: str, user_id: str, message: str):
        """Add an AI message to the conversation"""
        conversation = self.get_or_create_conversation(project_id, user_id)
        conversation['chat_history'].add_ai_message(message)
        conversation['last_activity'] = datetime.now()
        
        print(f"🤖 Added AI message to conversation {project_id}")
    
    def get_conversation_history(self, project_id: str, user_id: str) -> List[BaseMessage]:
        """Get the conversation history"""
        conversation = self.get_or_create_conversation(project_id, user_id)
        return conversation['chat_history'].get_messages()
    
    def get_memory(self, project_id: str, user_id: str, master_prompt: str = None):
        """Get the conversation memory object"""
        conversation = self.get_or_create_conversation(project_id, user_id, master_prompt)
        return conversation['memory']
    
    def clear_conversation(self, project_id: str, user_id: str):
        """Clear a specific conversation"""
        conv_key = self.get_conversation_key(project_id, user_id)
        
        if conv_key in self.conversations:
            self.conversations[conv_key]['chat_history'].clear()
            # Recreate memory to clear it
            if self.llm:
                try:
                    self.conversations[conv_key]['memory'] = ConversationSummaryBufferMemory(
                        llm=self.llm,
                        chat_memory=self.conversations[conv_key]['chat_history'],
                        max_token_limit=2000,
                        return_messages=True,
                        memory_key="chat_history"
                    )
                except:
                    from langchain.memory import ConversationBufferMemory
                    self.conversations[conv_key]['memory'] = ConversationBufferMemory(
                        chat_memory=self.conversations[conv_key]['chat_history'],
                        return_messages=True,
                        memory_key="chat_history"
                    )
            print(f"🗑️ Cleared conversation for project {project_id}")
        
        return {"message": "Conversation cleared successfully"}
    
    def cleanup_old_conversations(self, max_age_hours: int = 24):
        """Clean up old inactive conversations to free memory"""
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        keys_to_remove = []
        for key, conv in self.conversations.items():
            if conv['last_activity'] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.conversations[key]
        
        if keys_to_remove:
            print(f"🧹 Cleaned up {len(keys_to_remove)} old conversations")
    
    def get_conversation_stats(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Get statistics about the conversation"""
        conversation = self.get_or_create_conversation(project_id, user_id)
        messages = conversation['chat_history'].get_messages()
        
        user_messages = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        ai_messages = sum(1 for msg in messages if isinstance(msg, AIMessage))
        
        return {
            'total_messages': len(messages),
            'user_messages': user_messages,
            'ai_messages': ai_messages,
            'created_at': conversation['created_at'].isoformat(),
            'last_activity': conversation['last_activity'].isoformat()
        }

# Global conversation manager instance
conversation_manager = ConversationManager()