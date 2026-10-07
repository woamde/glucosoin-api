# Alias pour l'historique (utilisé par le frontend)
@app.get("/api/historique")
async def get_historique(limit: Optional[int] = None):
    return await get_journal(limit=limit)