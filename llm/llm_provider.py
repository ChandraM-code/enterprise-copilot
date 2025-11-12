from typing import Optional, List, Dict, Any, Callable
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.llms.base import LLM
from langchain.chat_models.base import BaseChatModel
import requests
from config import settings


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """Generate response for the given query"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of the LLM provider"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM Provider using Langchain"""
    
    def __init__(self, model: Optional[str] = None):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        self.model = model or settings.openai_model
        self.llm = ChatOpenAI(
            model=self.model,
            openai_api_key=settings.openai_api_key,
            temperature=0.7
        )
    
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """Generate response using OpenAI"""
        messages = []
        
        if context:
            messages.append(SystemMessage(content=f"Use the following context to answer the question:\n\n{context}"))
        
        messages.append(HumanMessage(content=query))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_provider_name(self) -> str:
        return f"OpenAI ({self.model})"


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM Provider using Langchain"""
    
    def __init__(self, model: Optional[str] = None):
        if not settings.google_api_key:
            raise ValueError("Google API key not configured")
        
        self.model = model or settings.gemini_model
        self.llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=settings.google_api_key,
            temperature=0.7
        )
    
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """Generate response using Gemini"""
        messages = []
        
        if context:
            messages.append(SystemMessage(content=f"Use the following context to answer the question:\n\n{context}"))
        
        messages.append(HumanMessage(content=query))
        
        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    def get_provider_name(self) -> str:
        return f"Google Gemini ({self.model})"


class CustomLLMProvider(BaseLLMProvider):
    """
    Custom LLM Provider - Supports multiple custom LLM types:
    1. Langchain LLM/ChatModel instances
    2. Custom API endpoints
    3. Custom callable functions
    """
    
    def __init__(
        self,
        llm_instance: Optional[Any] = None,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_function: Optional[Callable] = None,
        model_name: str = "Custom LLM",
        temperature: float = 0.7,
        headers: Optional[Dict] = None,
        **kwargs
    ):
        """
        Initialize custom LLM provider
        
        Args:
            llm_instance: A Langchain LLM or ChatModel instance
            api_endpoint: Custom API endpoint URL
            api_key: API key for custom endpoint
            custom_function: Custom function that takes (query, context) and returns response
            model_name: Name for the custom model
            temperature: Temperature setting
            headers: Additional headers for API requests
            **kwargs: Additional arguments for the LLM
        """
        self.model_name = model_name
        self.temperature = temperature
        self.kwargs = kwargs
        
        # Determine which type of custom LLM
        if llm_instance is not None:
            self.llm_type = "langchain"
            self.llm = llm_instance
        elif api_endpoint is not None:
            self.llm_type = "api"
            self.api_endpoint = api_endpoint
            self.api_key = api_key
            self.headers = headers or {}
            if api_key:
                self.headers["Authorization"] = f"Bearer {api_key}"
        elif custom_function is not None:
            self.llm_type = "function"
            self.custom_function = custom_function
        else:
            raise ValueError(
                "Must provide one of: llm_instance, api_endpoint, or custom_function"
            )
    
    def generate_response(self, query: str, context: Optional[str] = None) -> str:
        """Generate response using custom LLM"""
        try:
            if self.llm_type == "langchain":
                return self._generate_with_langchain(query, context)
            elif self.llm_type == "api":
                return self._generate_with_api(query, context)
            elif self.llm_type == "function":
                return self._generate_with_function(query, context)
        except Exception as e:
            raise Exception(f"Custom LLM error: {str(e)}")
    
    def _generate_with_langchain(self, query: str, context: Optional[str] = None) -> str:
        """Generate response using Langchain LLM instance"""
        # Check if it's a ChatModel (uses messages)
        if isinstance(self.llm, BaseChatModel):
            messages = []
            if context:
                messages.append(SystemMessage(
                    content=f"Use the following context to answer the question:\n\n{context}"
                ))
            messages.append(HumanMessage(content=query))
            response = self.llm.invoke(messages)
            return response.content
        else:
            # Regular LLM (uses text)
            prompt = query
            if context:
                prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            response = self.llm.invoke(prompt)
            return response if isinstance(response, str) else str(response)
    
    def _generate_with_api(self, query: str, context: Optional[str] = None) -> str:
        """Generate response using custom API endpoint"""
        prompt = query
        if context:
            prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            **self.kwargs
        }
        
        response = requests.post(
            self.api_endpoint,
            json=payload,
            headers=self.headers,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Try common response formats
        if isinstance(result, str):
            return result
        elif "response" in result:
            return result["response"]
        elif "text" in result:
            return result["text"]
        elif "output" in result:
            return result["output"]
        elif "choices" in result and len(result["choices"]) > 0:
            # OpenAI-like format
            choice = result["choices"][0]
            if "message" in choice:
                return choice["message"]["content"]
            elif "text" in choice:
                return choice["text"]
        
        # Return raw JSON if format is unknown
        return str(result)
    
    def _generate_with_function(self, query: str, context: Optional[str] = None) -> str:
        """Generate response using custom function"""
        response = self.custom_function(query, context)
        return response if isinstance(response, str) else str(response)
    
    def get_provider_name(self) -> str:
        return f"Custom ({self.model_name})"


class LLMManager:
    """Manager class to handle multiple LLM providers"""
    
    def __init__(self):
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers based on configuration"""
        if settings.openai_api_key:
            try:
                self.providers["openai"] = OpenAIProvider()
                print("✓ OpenAI provider initialized")
            except Exception as e:
                print(f"✗ Failed to initialize OpenAI: {e}")
        
        if settings.google_api_key:
            try:
                self.providers["gemini"] = GeminiProvider()
                print("✓ Gemini provider initialized")
            except Exception as e:
                print(f"✗ Failed to initialize Gemini: {e}")
    
    def register_custom_provider(
        self,
        provider_name: str,
        llm_instance: Optional[Any] = None,
        api_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_function: Optional[Callable] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Register a custom LLM provider
        
        Args:
            provider_name: Unique name for the provider (e.g., "my_custom_llm")
            llm_instance: A Langchain LLM or ChatModel instance
            api_endpoint: Custom API endpoint URL
            api_key: API key for custom endpoint
            custom_function: Custom function that takes (query, context) and returns response
            model_name: Display name for the model
            **kwargs: Additional arguments passed to CustomLLMProvider
        
        Examples:
            # Using a Langchain instance
            manager.register_custom_provider(
                "llama",
                llm_instance=ChatOllama(model="llama2")
            )
            
            # Using a custom API endpoint
            manager.register_custom_provider(
                "my_api",
                api_endpoint="https://api.example.com/generate",
                api_key="your-api-key"
            )
            
            # Using a custom function
            def my_llm(query, context):
                return f"Response to: {query}"
            
            manager.register_custom_provider(
                "my_function",
                custom_function=my_llm
            )
        """
        try:
            custom_provider = CustomLLMProvider(
                llm_instance=llm_instance,
                api_endpoint=api_endpoint,
                api_key=api_key,
                custom_function=custom_function,
                model_name=model_name or provider_name,
                **kwargs
            )
            self.providers[provider_name] = custom_provider
            print(f"✓ Custom provider '{provider_name}' registered")
        except Exception as e:
            print(f"✗ Failed to register custom provider '{provider_name}': {e}")
            raise
    
    def unregister_provider(self, provider_name: str) -> None:
        """Remove a provider from the manager"""
        if provider_name in self.providers:
            del self.providers[provider_name]
            print(f"✓ Provider '{provider_name}' unregistered")
        else:
            print(f"⚠️  Provider '{provider_name}' not found")
    
    def get_provider(self, provider_name: Optional[str] = None) -> BaseLLMProvider:
        """Get LLM provider by name"""
        provider_name = provider_name or settings.default_llm
        
        if provider_name not in self.providers:
            available = ", ".join(self.providers.keys())
            raise ValueError(f"Provider '{provider_name}' not available. Available providers: {available}")
        
        return self.providers[provider_name]
    
    def generate_response(self, query: str, context: Optional[str] = None, provider_name: Optional[str] = None) -> Dict:
        """Generate response using specified provider"""
        provider = self.get_provider(provider_name)
        
        print(f"🤖 Calling LLM: {provider.get_provider_name()}")
        response = provider.generate_response(query, context)
        
        return {
            "response": response,
            "provider": provider.get_provider_name()
        }
    
    def list_providers(self) -> List[str]:
        """List all available providers"""
        return list(self.providers.keys())

