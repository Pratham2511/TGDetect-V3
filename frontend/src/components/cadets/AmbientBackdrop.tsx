import styles from "./ambient.module.css";

/** Decorative motion stays behind the opaque data panels and graph. */
export function AmbientBackdrop() {
  return (
    <div className={styles.backdrop} aria-hidden="true">
      <div className={styles.bloom} />
      <svg className={styles.waves} viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid slice" fill="none">
        <defs>
          <linearGradient id="tgd-wine-ribbon" x1="450" y1="30" x2="1570" y2="910" gradientUnits="userSpaceOnUse">
            <stop stopColor="#280911" stopOpacity="0" />
            <stop offset=".42" stopColor="#6e2039" stopOpacity=".3" />
            <stop offset=".7" stopColor="#ba6974" stopOpacity=".18" />
            <stop offset="1" stopColor="#370d1a" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="tgd-wine-edge" x1="500" y1="90" x2="1540" y2="750" gradientUnits="userSpaceOnUse">
            <stop stopColor="#613044" stopOpacity="0" />
            <stop offset=".52" stopColor="#a45b6b" stopOpacity=".28" />
            <stop offset="1" stopColor="#662339" stopOpacity="0" />
          </linearGradient>
        </defs>
        <g className={styles.ribbon}>
          <path d="M590-130C1010-40 1540 95 1520 340C1500 550 1020 475 1120 740C1160 865 1480 910 1730 1040" stroke="url(#tgd-wine-ribbon)" strokeWidth="120" />
          <path d="M615-120C1040-10 1510 100 1498 340C1485 557 1015 482 1100 745C1148 890 1490 931 1720 1060" stroke="url(#tgd-wine-edge)" strokeWidth="1.5" />
          <path d="M710-80C1120 12 1460 155 1440 350C1418 560 982 510 1060 772C1100 925 1420 979 1610 1100" stroke="url(#tgd-wine-edge)" strokeWidth=".75" />
        </g>
        <g className={styles.glints} fill="#d4a3a8">
          <circle cx="1080" cy="112" r=".8" />
          <circle cx="1375" cy="208" r="1" />
          <circle cx="1520" cy="465" r=".65" />
          <circle cx="1450" cy="640" r=".8" />
          <circle cx="1230" cy="810" r=".65" />
          <circle cx="1505" cy="850" r="1" />
          <circle cx="930" cy="78" r=".6" />
          <circle cx="1300" cy="960" r=".7" />
        </g>
      </svg>
    </div>
  );
}
