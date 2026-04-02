import { useMemo, useState, type ReactNode } from 'react'
import { LanguageContext, type Lang } from './i18n-context'

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>('cn')

  const value = useMemo(
    () => ({
      lang,
      toggle: () => setLang((current) => (current === 'cn' ? 'en' : 'cn')),
      setLang,
    }),
    [lang],
  )

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}
