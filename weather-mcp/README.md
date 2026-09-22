```bash
# 方式一（推荐）：uv run，在仓库根目录执行                                                           
uv run mcp dev src/weather_mcp/server.py                                                             
                                                                                                       
# 方式二：激活 venv 后再跑                                                                           
source .venv/bin/activate                                                                            
mcp dev src/weather_mcp/server.py                                                                    
                                                                                                        
# 方式三：直接用 venv 里的可执行文件                                                                 
.venv/bin/mcp dev src/weather_mcp/server.py                                                          
```

