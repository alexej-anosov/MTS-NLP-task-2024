from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict
)


class Config(BaseSettings):

    BUCKET_HOST: str
    BUCKET_KEY_ID: str
    BUCKET_KEY: str
    BUCKET_SERVICE: str
    BUCKET_NAME: str
    
    OPENAI_API_KEY: str
    PROXY_URL: str
    
    model_config = SettingsConfigDict(env_file = '.env', extra='ignore')
    
    
config = Config()
