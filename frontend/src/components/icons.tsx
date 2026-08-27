type IconProps = { size?: number };

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export function IconHome({ size = 17 }: IconProps) {
  return (
    <svg {...base(size)}>
      <path d="M3 11.5 12 4l9 7.5" />
      <path d="M5.5 10v9a1 1 0 0 0 1 1H9a1 1 0 0 0 1-1v-4h4v4a1 1 0 0 0 1 1h2.5a1 1 0 0 0 1-1v-9" />
    </svg>
  );
}

export function IconFlask({ size = 17 }: IconProps) {
  return (
    <svg {...base(size)}>
      <path d="M9 3h6" />
      <path d="M10 3v6.5L4.8 18a1.8 1.8 0 0 0 1.55 2.7h11.3A1.8 1.8 0 0 0 19.2 18L14 9.5V3" />
      <path d="M7.5 15h9" />
    </svg>
  );
}

export function IconChart({ size = 17 }: IconProps) {
  return (
    <svg {...base(size)}>
      <path d="M4 20V10" />
      <path d="M11 20V4" />
      <path d="M18 20v-7" />
      <path d="M3 20h18" />
    </svg>
  );
}

export function IconSun({ size = 15 }: IconProps) {
  return (
    <svg {...base(size)}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </svg>
  );
}

export function IconMoon({ size = 15 }: IconProps) {
  return (
    <svg {...base(size)}>
      <path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z" />
    </svg>
  );
}

export function IconNetwork({ size = 18 }: IconProps) {
  return (
    <svg {...base(size)} stroke="white">
      <circle cx="6" cy="6" r="2.3" />
      <circle cx="18" cy="6" r="2.3" />
      <circle cx="12" cy="13" r="2.3" />
      <circle cx="6" cy="20" r="2.3" />
      <circle cx="18" cy="20" r="2.3" />
      <path d="M7.9 7.3 10.3 11.3M16.1 7.3 13.7 11.3M10.6 14.7 7.4 18.3M13.4 14.7 16.6 18.3" />
    </svg>
  );
}

export function IconScales({ size = 18 }: IconProps) {
  return (
    <svg {...base(size)} stroke="white">
      <path d="M12 3v18M8 21h8" />
      <path d="M4 7h6M14 7h6" />
      <path d="M4 7 1.5 12.5a2.8 2.8 0 0 0 5 0Z" />
      <path d="M20 7l-2.5 5.5a2.8 2.8 0 0 0 5 0Z" />
    </svg>
  );
}

export function IconLayers({ size = 18 }: IconProps) {
  return (
    <svg {...base(size)} stroke="white">
      <path d="M12 3 3 8l9 5 9-5-9-5Z" />
      <path d="M3 13l9 5 9-5" />
    </svg>
  );
}
