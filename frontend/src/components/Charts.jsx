import React from 'react';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';

const SEMANTIC = {
  low: '#12B2C1',
  medium: '#FFB162',
  high: '#FF4D4D',
};

const getLevelByPercent = (value) => {
  if (!Number.isFinite(value)) return 'medium';
  if (value >= 35) return 'high';
  if (value >= 18) return 'medium';
  return 'low';
};

const getDonutColor = (entry, index) => {
  const name = String(entry?.name || '').toLowerCase();
  if (name.includes('churn')) return SEMANTIC.high;
  if (name.includes('retain') || name.includes('stable') || name.includes('safe')) return SEMANTIC.low;
  return index % 2 === 0 ? SEMANTIC.low : SEMANTIC.medium;
};

export const DonutChart = ({ data, title }) => (
  <div className="glow-card backdrop-blur-sm p-6 rounded-2xl shadow-xl h-[400px] flex flex-col">
    <h3 className="text-[#E5F9F8] font-bold mb-6 flex items-center gap-2">
      <span className="h-2 w-2 rounded-full bg-[#12B2C1]"></span>
      {title}
    </h3>
    <div className="flex-grow">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            innerRadius={60}
            outerRadius={80}
            paddingAngle={5}
            dataKey="value"
            stroke="none"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getDonutColor(entry, index)} />
            ))}
          </Pie>
          <Tooltip 
            contentStyle={{ backgroundColor: '#05080F', borderColor: '#23717B', borderRadius: '12px', color: '#E5F9F8' }}
            itemStyle={{ color: '#E5F9F8' }}
          />
          <Legend 
            verticalAlign="bottom" 
            height={36} 
            formatter={(value) => <span className="text-[#A8CFCF] text-xs font-medium uppercase tracking-wider">{value}</span>}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  </div>
);

export const ComparisonBarChart = ({ data, title, xKey, yKey }) => (
  <div className="glow-card backdrop-blur-sm p-6 rounded-2xl shadow-xl h-[400px] flex flex-col">
    <h3 className="text-[#E5F9F8] font-bold mb-6 flex items-center gap-2">
      <span className="h-2 w-2 rounded-full bg-[#0D8A9E]"></span>
      {title}
    </h3>
    <div className="flex-grow">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1A3B40" vertical={false} />
          <XAxis 
            dataKey={xKey} 
            stroke="#A8CFCF" 
            fontSize={12} 
            tickLine={false} 
            axisLine={false}
          />
          <YAxis 
            stroke="#A8CFCF" 
            fontSize={12} 
            tickLine={false} 
            axisLine={false}
            tickFormatter={(value) => `${value}%`}
          />
          <Tooltip 
            cursor={{ fill: '#1A3B40', opacity: 0.35 }}
            contentStyle={{ backgroundColor: '#05080F', borderColor: '#23717B', borderRadius: '12px', color: '#E5F9F8' }}
          />
          <Bar dataKey={yKey} radius={[4, 4, 0, 0]} barSize={40}>
            {data.map((entry, index) => {
              const level = getLevelByPercent(Number(entry?.[yKey]));
              const fill = SEMANTIC[level];
              return (
                <Cell
                  key={`bar-cell-${index}`}
                  fill={fill}
                  stroke={fill}
                  strokeWidth={1.2}
                  style={{ filter: `drop-shadow(0 0 7px ${fill}66)` }}
                />
              );
            })}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);
