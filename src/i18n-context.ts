import { createContext, useContext } from 'react'

export type Lang = 'cn' | 'en'

type LanguageContextValue = {
  lang: Lang
  toggle: () => void
  setLang: (lang: Lang) => void
}

export const LanguageContext = createContext<LanguageContextValue | null>(null)

export function useLanguage() {
  const context = useContext(LanguageContext)
  if (!context) {
    throw new Error('useLanguage must be used within LanguageProvider')
  }
  return context
}
