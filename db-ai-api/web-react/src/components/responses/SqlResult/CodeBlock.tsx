import React, { useState } from 'react';

interface CodeBlockProps {
  code: string;
  language?: string;
  showLineNumbers?: boolean;
  maxHeight?: string;
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code = '',
  language = 'sql',
  showLineNumbers = true,
  maxHeight = '300px',
}) => {
  const [copied, setCopied] = useState(false);
  const safeCode = code || '';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(safeCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines = safeCode.split('\n');

  const highlightSQL = (line: string): React.ReactNode => {
    const keywords =
      /\b(SELECT|FROM|WHERE|AND|OR|NOT|IN|LIKE|BETWEEN|JOIN|LEFT|RIGHT|INNER|OUTER|ON|GROUP|BY|ORDER|ASC|DESC|HAVING|LIMIT|OFFSET|UNION|ALL|DISTINCT|AS|COUNT|SUM|AVG|MIN|MAX|CASE|WHEN|THEN|ELSE|END|NULL|IS|TOP|WITH|OVER|PARTITION|ROW_NUMBER|COALESCE|ISNULL|CAST|CONVERT)\b/gi;
    const strings = /('[^']*')/g;
    const comments = /(--.*$)/gm;

    const result = line
      .replace(comments, '<span class="text-slate-500 italic">$1</span>')
      .replace(strings, '<span class="text-emerald-300">$1</span>')
      .replace(keywords, '<span class="text-sky-300 font-semibold">$1</span>')
      .replace(/\b(\d+\.?\d*)\b/g, '<span class="text-amber-300">$1</span>');

    return <span dangerouslySetInnerHTML={{ __html: result }} />;
  };

  return (
    <div className="code-block relative rounded-xl overflow-hidden border border-white/10 bg-[#0a0f16]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-white/5 border-b border-white/10">
        <span className="text-xs font-medium text-slate-400 uppercase tracking-[0.2em]">
          {language}
        </span>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-1 text-xs text-slate-400 hover:text-white hover:bg-white/10 rounded transition-colors"
        >
          {copied ? (
            <>
              <svg className="w-4 h-4 text-emerald-300 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path d="M5 13l4 4L19 7" />
              </svg>
              Copied
            </>
          ) : (
            <>
              <svg className="w-4 h-4 thin-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <rect x="9" y="9" width="11" height="11" rx="2" />
                <rect x="4" y="4" width="11" height="11" rx="2" />
              </svg>
              Copy
            </>
          )}
        </button>
      </div>

      {/* Code content */}
      <div
        className="overflow-auto p-4 font-mono text-sm text-slate-100"
        style={{ maxHeight }}
      >
        <table className="w-full">
          <tbody>
            {lines.map((line, index) => (
              <tr key={index} className="leading-relaxed">
                {showLineNumbers && (
                  <td className="pr-4 text-right text-slate-600 select-none w-8">
                    {index + 1}
                  </td>
                )}
                <td className="whitespace-pre">
                  {highlightSQL(line)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default CodeBlock;
